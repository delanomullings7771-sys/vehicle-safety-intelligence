const API = (window.API_BASE || "").replace(/\/$/, "");
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v, d = 1) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
const num = (v) => (v == null ? "—" : Number(v).toLocaleString());

async function api(path, opts = {}) {
  const res = await fetch(API + path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) {}
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return res.json();
}

function showError(el, err) { el.innerHTML = `<div class="banner">${esc(err.message || err)}</div>`; }

// Tabs
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", x === b));
  document.querySelectorAll(".panel").forEach((p) => (p.hidden = p.id !== b.dataset.tab));
  history.replaceState(null, "", "#" + b.dataset.tab);
}));
const openFromHash = () => {
  const tab = location.hash.slice(1);
  if (tab) document.querySelector(`.tabs button[data-tab="${tab}"]`)?.click();
};
window.addEventListener("hashchange", openFromHash);
openFromHash();

// Overview metrics
(async () => {
  try {
    const m = await api("/api/meta");
    $("#disclaimer").textContent = m.disclaimer;
    $("#metrics").innerHTML = m.models.map((x) => `
      <div class="metric"><div class="k">${esc(x.task)}</div>
        <div class="v">${esc(x.headline_value)}</div>
        <div class="d">${esc(x.headline_metric)} · ${esc(x.model)}<br>${esc(x.detail || "")}</div></div>`).join("");
  } catch (e) {
    const b = $("#api-status"); b.hidden = false;
    b.textContent = `Cannot reach the model API at ${API || "this origin"}. It may be waking up (free hosting sleeps when idle); retry in a minute.`;
    $("#metrics").innerHTML = "";
  }
})();

// Complaint triage
const EXAMPLE = "WHILE DRIVING ON THE HIGHWAY AT ABOUT 60 MPH THE CAR SUDDENLY LOST POWER AND THE CHECK ENGINE LIGHT CAME ON. " +
  "THE STEERING BECAME VERY HEAVY AND I COULD NOT CONTROL THE VEHICLE. I HIT THE GUARDRAIL AND THE AIR BAGS DID NOT DEPLOY. " +
  "MY PASSENGER WAS TAKEN TO THE HOSPITAL. A POLICE REPORT WAS FILED. THE DEALER SAID THE ENGINE HAD A KNOWN ISSUE.";
$("#example-btn").addEventListener("click", () => { $("#narrative").value = EXAMPLE; });
$("#complaint-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const out = $("#complaint-result"), btn = e.submitter; btn.disabled = true; out.innerHTML = '<p class="muted">Analysing…</p>';
  try {
    const r = await api("/api/complaint/analyze", { method: "POST", body: JSON.stringify({ text: $("#narrative").value }) });
    const maxS = Math.max(...r.components.map((c) => c.score - c.threshold), 0.001);
    const s = r.serious_incident;
    out.innerHTML = `
      <div class="box"><h3 style="margin-top:0">Serious-incident flag
        <span class="flag ${s.flagged ? "on" : "off"}">${s.flagged ? "Flag for priority review" : "Not flagged"}</span></h3>
        <p class="muted">Score ${s.score.toFixed(3)} against threshold ${s.threshold.toFixed(3)} · ${esc(r.models.serious_flag)}</p>
        ${s.evidence.length ? `<div class="chips">${s.evidence.map((t) => `<span class="chip">${esc(t.term)}</span>`).join("")}</div>` : ""}
      </div>
      <div class="box"><h3 style="margin-top:0">Recommended component groups</h3>
        <p class="muted">Highlighted groups pass their validated threshold · ${esc(r.models.routing)}</p>
        <div class="bars">${r.components.map((c) => {
          const w = Math.max(4, Math.min(100, ((c.score - c.threshold) / maxS) * 50 + 50));
          return `<div class="bar ${c.recommended ? "rec" : ""}"><span>${esc(c.component)}${c.recommended ? " ✓" : ""}</span>
            <span class="track"><span class="fill" style="width:${w}%; display:block"></span></span>
            <span class="muted">${c.score.toFixed(2)}</span></div>
            ${c.recommended && c.evidence.length ? `<div class="chips" style="margin:-2px 0 6px">${c.evidence.slice(0, 6).map((t) => `<span class="chip">${esc(t.term)}</span>`).join("")}</div>` : ""}`;
        }).join("")}</div>
      </div>
      <p class="disclaimer">${esc(r.disclaimer)}</p>`;
  } catch (err) { showError(out, err); } finally { btn.disabled = false; }
});

// Crash severity
let crashForm = null;
(async () => {
  try {
    crashForm = await api("/api/crash/form");
    $("#crash-fields").innerHTML = crashForm.fields.map((f) => {
      const help = f.help ? ` <span class="help">${esc(f.help)}</span>` : "";
      if (f.type === "number") {
        return `<div><label for="f-${f.name}">${esc(f.label)}${help}</label>
          <input id="f-${f.name}" name="${f.name}" type="number" min="${f.min ?? ""}" max="${f.max ?? ""}" step="1" placeholder="unknown"></div>`;
      }
      return `<div><label for="f-${f.name}">${esc(f.label)}${help}</label><select id="f-${f.name}" name="${f.name}">
        <option value="">Unknown / not stated</option>${f.options.map((o) => `<option value="${o.value}">${esc(o.label)}</option>`).join("")}
        </select></div>`;
    }).join("");
  } catch (e) { showError($("#crash-fields"), e); }
})();
$("#crash-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const out = $("#crash-result"), btn = e.submitter; btn.disabled = true; out.innerHTML = '<p class="muted">Assessing…</p>';
  const body = {};
  crashForm.fields.forEach((f) => { const v = $(`#f-${f.name}`).value; body[f.name] = v === "" ? null : Number(v); });
  try {
    const r = await api("/api/crash/assess", { method: "POST", body: JSON.stringify(body) });
    const card = (title, x) => `<div class="box"><h3 style="margin-top:0">${title}
        <span class="flag ${x.flagged ? "on" : "off"}">${x.flagged ? "Above review threshold" : "Below review threshold"}</span></h3>
        <div class="kv"><div><b>${pct(x.probability)}</b><span>model probability</span></div>
        <div><b>${pct(x.threshold)}</b><span>validated threshold</span></div></div>
        ${x.factors ? `<h3>Main factors in this estimate</h3><div class="bars">${x.factors.map((f) => `
          <div class="bar ${f.effect > 0 ? "rec" : ""}"><span>${esc(f.label)}</span>
          <span class="track"><span class="fill" style="width:${Math.min(100, Math.abs(f.effect) * 100)}%;display:block"></span></span>
          <span class="muted">${f.effect > 0 ? "raises" : "lowers"}</span></div>`).join("")}</div>` : ""}
        <p class="muted" style="margin-top:8px">${esc(x.model)} · ${esc(x.note)}</p></div>`;
    out.innerHTML = (r.warning ? `<div class="banner">${esc(r.warning)}</div>` : "") +
      card("Injury crash", r.injury_crash) + card("Serious or fatal injury", r.serious_or_fatal_crash) +
      `<p class="disclaimer">${esc(r.disclaimer)}</p>`;
  } catch (err) { showError(out, err); } finally { btn.disabled = false; }
});

// Vehicle lookup
const fill = (sel, items, ph) => { sel.innerHTML = `<option value="">${ph}</option>` + items.map((i) => `<option>${esc(i)}</option>`).join(""); };
(async () => {
  try {
    fill($("#make"), await api("/api/vehicles/makes"), "Choose a make");
    const top = await api("/api/vehicles/top?limit=25");
    $("#top-table tbody").innerHTML = top.map((t) => `<tr><td>${esc(t.model_year)} ${esc(t.make)} ${esc(t.model)}</td>
      <td class="num">${num(t.complaints)}</td><td class="num">${t.complaints_per_10k_crash_involved.toFixed(1)}</td>
      <td class="num">${pct(t.serious_share)}</td></tr>`).join("");
  } catch (e) { /* banner already shown by meta call */ }
})();
$("#make").addEventListener("change", async () => {
  const make = $("#make").value; $("#year").disabled = true; fill($("#year"), [], "—");
  if (!make) { $("#model").disabled = true; return; }
  fill($("#model"), await api(`/api/vehicles/models?make=${encodeURIComponent(make)}`), "Choose a model"); $("#model").disabled = false;
});
$("#model").addEventListener("change", async () => {
  const make = $("#make").value, model = $("#model").value;
  if (!model) { $("#year").disabled = true; return; }
  fill($("#year"), await api(`/api/vehicles/years?make=${encodeURIComponent(make)}&model=${encodeURIComponent(model)}`), "Choose a year");
  $("#year").disabled = false;
});
$("#year").addEventListener("change", async () => {
  const out = $("#vehicle-result"), y = $("#year").value; if (!y) return;
  out.innerHTML = '<p class="muted">Loading…</p>';
  try {
    const r = await api(`/api/vehicles/profile?make=${encodeURIComponent($("#make").value)}&model=${encodeURIComponent($("#model").value)}&year=${y}`);
    const c = r.complaints, x = r.crash_exposure, g = r.integrated;
    const comps = Object.entries(c.by_component), maxC = Math.max(...comps.map((kv) => kv[1]), 1);
    out.innerHTML = `
      <div class="box"><h3 style="margin-top:0">${esc(r.vehicle.model_year)} ${esc(r.vehicle.make)} ${esc(r.vehicle.model)}</h3>
        <div class="kv">
          <div><b>${num(c.total_2020_2024)}</b><span>complaints 2020–2024</span></div>
          <div><b>${pct(c.serious_share)}</b><span>report a crash, fire, injury or death</span></div>
          <div><b>${g.reliable ? g.complaints_per_10k_crash_involved.toFixed(1) : "—"}</b><span>complaints per 10k crash-involved vehicles</span></div>
          <div><b>${g.reliable ? Math.round(g.percentile_among_vehicles * 100) + "th" : "—"}</b><span>percentile among comparable vehicles</span></div>
        </div>
        ${g.reliable ? "" : '<p class="muted">Too few CRSS crash records for this vehicle to estimate exposure reliably.</p>'}
      </div>
      <div class="box"><h3 style="margin-top:0">Crash record (CRSS, survey-weighted)</h3>
        <div class="kv">
          <div><b>${num(x.estimated_crash_involved_vehicles && Math.round(x.estimated_crash_involved_vehicles))}</b><span>estimated crash-involved vehicles, 2020–2024</span></div>
          <div><b>${pct(x.injury_crash_rate)}</b><span>of those crashes involved injury</span></div>
          <div><b>${pct(x.serious_crash_rate)}</b><span>involved serious or fatal injury</span></div>
          <div><b>${pct(x.defect_recorded_rate, 2)}</b><span>had a police-recorded vehicle defect</span></div>
        </div></div>
      <div class="box"><h3 style="margin-top:0">Complaints by component</h3><div class="bars">
        ${comps.map(([k, v]) => `<div class="bar"><span>${esc(k)}</span><span class="track"><span class="fill" style="width:${(v / maxC) * 100}%;display:block"></span></span><span class="muted">${num(v)}</span></div>`).join("")}
      </div></div>
      <p class="disclaimer">${esc(r.disclaimer)}</p>`;
  } catch (err) { showError(out, err); }
});
