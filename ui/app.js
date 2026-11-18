// PathSnap UI, runs on Laptop 2.
// It only calls the API: GET /health, GET /trips, POST /predict.
// Everything shown comes from the /predict response. The reference route
// and labels stay on the server and are never sent here.

const $ = (id) => document.getElementById(id);

// ---------- API address ----------
function initialApiUrl() {
  const fromQuery = new URLSearchParams(location.search).get("api");
  if (fromQuery) return fromQuery;
  try {
    const saved = localStorage.getItem("pathsnap_api");
    if (saved) return saved;
  } catch (e) { /* storage may be blocked */ }
  return window.PATHSNAP_API_DEFAULT || "http://127.0.0.1:8000";
}

let apiUrl = initialApiUrl().replace(/\/+$/, "");
$("api-url").value = apiUrl;

// ---------- fetch with timeout and readable errors ----------
async function callApi(path, options = {}, timeoutS = 10) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutS * 1000);
  let res;
  try {
    res = await fetch(apiUrl + path, { ...options, signal: controller.signal });
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error(`No reply from the API within ${timeoutS} s. The server may be busy or stuck. Check its console on Laptop 1.`);
    }
    throw new Error(`Cannot reach the API at ${apiUrl}. Check that it is running, both laptops are on the same network, and the API port is allowed through Laptop 1's firewall.`);
  } finally {
    clearTimeout(timer);
  }
  let body = null;
  try { body = await res.json(); } catch (e) { /* not JSON */ }
  if (!res.ok) {
    const detail = body && body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : res.statusText;
    throw new Error(`API error ${res.status}: ${detail}`);
  }
  return { body, pipeline: res.headers.get("X-PathSnap-Pipeline") };
}

function setStatus(el, text, kind = "") {
  el.className = "status " + kind;
  el.innerHTML = text;
}

// ---------- connect and list trips ----------
async function connect() {
  apiUrl = $("api-url").value.trim().replace(/\/+$/, "");
  try { localStorage.setItem("pathsnap_api", apiUrl); } catch (e) { /* ignore */ }
  setStatus($("api-status"), '<span class="spinner"></span>Connecting...');
  $("trips").innerHTML = "";
  try {
    const health = await callApi("/health");
    if (!health.body || health.body.status !== "ok") throw new Error("Unexpected /health reply");
    const { body } = await callApi("/trips");
    setStatus($("api-status"), `Connected to ${apiUrl}`, "ok");
    showPlaceholderBanner(health.pipeline);
    renderTrips(body.trips || []);
  } catch (err) {
    setStatus($("api-status"), err.message, "err");
    $("trips").innerHTML = '<p class="muted">No trips loaded.</p>';
  }
}

function renderTrips(trips) {
  const box = $("trips");
  box.innerHTML = "";
  if (!trips.length) {
    box.innerHTML = '<p class="muted">The API returned no trips.</p>';
    return;
  }
  for (const id of trips) {
    const b = document.createElement("button");
    b.textContent = id;
    b.dataset.trip = id;
    b.onclick = () => runTrip(id);
    box.appendChild(b);
  }
}

function showPlaceholderBanner(pipeline) {
  $("placeholder-banner").hidden = pipeline !== "placeholder";
}

// ---------- prediction ----------
let currentRequest = 0;
let lastTrip = null;

async function runTrip(tripId) {
  const requestId = ++currentRequest;  // a newer click makes older replies stale
  lastTrip = tripId;
  document.querySelectorAll(".trips button").forEach((b) => b.classList.toggle("active", b.dataset.trip === tripId));
  $("retry").hidden = true;
  $("metrics").hidden = true;
  $("extra-metrics").textContent = "";
  clearLayers();

  const started = performance.now();
  const showElapsed = () => {
    const s = ((performance.now() - started) / 1000).toFixed(1);
    setStatus($("run-status"), `<span class="spinner"></span>Running trip ${tripId} on the API... ${s} s`);
  };
  showElapsed();
  const tick = setInterval(showElapsed, 100);

  try {
    const { body, pipeline } = await callApi("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trip_id: tripId }),
    }, window.PATHSNAP_TIMEOUT_S || 60);
    if (requestId !== currentRequest) return;
    const roundTrip = ((performance.now() - started) / 1000).toFixed(2);
    showPlaceholderBanner(pipeline);
    setStatus($("run-status"), `Trip ${body.trip_id} done. Round trip ${roundTrip} s.`, "ok");
    showResult(body);
  } catch (err) {
    if (requestId !== currentRequest) return;
    setStatus($("run-status"), err.message, "err");
    $("retry").hidden = false;
  } finally {
    clearInterval(tick);
  }
}

$("retry").onclick = () => lastTrip && runTrip(lastTrip);
$("connect").onclick = connect;
$("api-url").addEventListener("keydown", (e) => { if (e.key === "Enter") connect(); });

// ---------- metrics ----------
const pct = (x) => (x === null || x === undefined ? "n/a" : (x * 100).toFixed(1) + "%");

function showResult(r) {
  const m = r.metrics || {};
  const matched = r.predicted.filter((p) => p.arc_id >= 0).length;
  $("m-top1").textContent = pct(m.top1_accuracy);
  $("m-cont").textContent = pct(m.route_continuity);
  $("m-time").textContent = m.inference_ms === undefined || m.inference_ms === null ? "n/a" : `${Math.round(m.inference_ms)} ms`;
  $("m-points").textContent = `${matched} / ${r.gps.length}`;
  $("metrics").hidden = false;
  const extras = [];
  if (m.macro_f1 !== undefined && m.macro_f1 !== null) extras.push(`Macro F1 ${pct(m.macro_f1)}`);
  if (m.candidate_miss_rate !== undefined && m.candidate_miss_rate !== null) extras.push(`Candidate miss rate ${pct(m.candidate_miss_rate)}`);
  $("extra-metrics").textContent = extras.join("  |  ");
  drawMap(r);
}

// ---------- map ----------
const map = L.map("map", { preferCanvas: true }).setView([56.0, 37.0], 5);
const tiles = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

const layers = {
  gps: L.layerGroup().addTo(map),
  route: L.layerGroup().addTo(map),
  ends: L.layerGroup().addTo(map),
};
L.control.layers({ "OpenStreetMap": tiles }, {
  "GPS points": layers.gps,
  "Predicted route": layers.route,
  "Start / end": layers.ends,
}, { collapsed: false }).addTo(map);
L.control.scale().addTo(map);

function clearLayers() {
  Object.values(layers).forEach((g) => g.clearLayers());
}

function drawMap(r) {
  clearLayers();
  const arcByObs = new Map(r.predicted.map((p) => [p.obs_idx, p.arc_id]));

  r.gps.forEach(([lat, lon], i) => {
    const arc = arcByObs.has(i) ? arcByObs.get(i) : -1;
    L.circleMarker([lat, lon], {
      radius: 3, weight: 0, fillOpacity: 0.75,
      fillColor: arc >= 0 ? "#2b59c3" : "#f79009",
    }).bindTooltip(`point ${i}<br>arc ${arc >= 0 ? arc : "none"}`).addTo(layers.gps);
  });

  if (r.route.length > 1) {
    L.polyline(r.route, { color: "#d92d20", weight: 4, opacity: 0.85 }).addTo(layers.route);
  }

  if (r.gps.length) {
    const first = r.gps[0], last = r.gps[r.gps.length - 1];
    L.circleMarker(first, { radius: 8, color: "#fff", weight: 2, fillColor: "#12b76a", fillOpacity: 1 }).bindTooltip("Start").addTo(layers.ends);
    L.circleMarker(last, { radius: 8, color: "#fff", weight: 2, fillColor: "#101828", fillOpacity: 1 }).bindTooltip("End").addTo(layers.ends);
    map.fitBounds(L.latLngBounds(r.gps.concat(r.route)), { padding: [30, 30] });
  }
}

connect();
