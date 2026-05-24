/* kernel-synth UI */
(() => {
  const repoListEl = document.getElementById("repo-list");
  const repoCountEl = document.getElementById("repo-count");
  const contentEl = document.getElementById("content");
  const searchEl = document.getElementById("search");
  const topbarStatsEl = document.getElementById("topbar-stats");
  const tabsEl = document.getElementById("tabs");
  const layoutReposEl = document.getElementById("layout-repos");
  const layoutEnvsEl = document.getElementById("layout-envs");
  const envsContentEl = document.getElementById("envs-content");

  const state = {
    repos: [],
    stats: null,
    health: null,
    activeSlug: null,
    activeRecord: null,
    filter: "",
    envs: null,
    activeTab: "repos",
    envTagFilter: null, // single active chip name, or null = show all
  };

  // -----------------------------------------------------------
  // Boot
  // -----------------------------------------------------------
  async function boot() {
    await Promise.all([loadRepos(), loadStats(), loadHealth()]);
    renderTopbar();
    renderSidebar();
    if (state.repos.length > 0) {
      const fromHash = decodeURIComponent((location.hash || "").replace(/^#/, ""));
      const initial = state.repos.find((r) => r.slug === fromHash) || state.repos[0];
      selectRepo(initial.slug);
    }
    if (tabsEl) {
      tabsEl.addEventListener("click", (e) => {
        const btn = e.target.closest("button[data-tab]");
        if (!btn) return;
        switchTab(btn.dataset.tab);
      });
    }
  }

  function switchTab(tab) {
    state.activeTab = tab;
    for (const btn of tabsEl.querySelectorAll("button[data-tab]")) {
      btn.classList.toggle("active", btn.dataset.tab === tab);
    }
    if (tab === "envs") {
      layoutReposEl.classList.add("hidden");
      layoutEnvsEl.classList.remove("hidden");
      if (state.envs === null) loadAndRenderEnvs();
      else renderEnvs(state.envs);
    } else {
      layoutEnvsEl.classList.add("hidden");
      layoutReposEl.classList.remove("hidden");
    }
  }

  searchEl.addEventListener("input", (e) => {
    state.filter = e.target.value.trim().toLowerCase();
    renderSidebar();
    if (state.activeRecord) renderRepoDetail(state.activeRecord);
  });

  window.addEventListener("hashchange", () => {
    const slug = decodeURIComponent((location.hash || "").replace(/^#/, ""));
    if (slug && slug !== state.activeSlug) selectRepo(slug);
  });

  // ----- keyboard navigation -----
  // j/k move repo selection one row at a time (vim-style), "/" focuses the
  // search box, Escape clears the search and returns focus to the body.
  // We only intercept these when no input/textarea has focus so the keys
  // remain usable inside search.
  document.addEventListener("keydown", (e) => {
    const active = document.activeElement;
    const inEditable =
      active &&
      ((active.tagName === "INPUT" && active.type !== "button") ||
        active.tagName === "TEXTAREA" ||
        active.isContentEditable);
    if (e.key === "/" && !inEditable) {
      e.preventDefault();
      searchEl.focus();
      searchEl.select();
      return;
    }
    if (e.key === "Escape" && active === searchEl) {
      searchEl.value = "";
      state.filter = "";
      renderSidebar();
      if (state.activeRecord) renderRepoDetail(state.activeRecord);
      searchEl.blur();
      return;
    }
    if (inEditable) return;
    if (state.activeTab !== "repos") return;
    if (e.key === "j" || e.key === "k") {
      const visible = state.repos.filter((r) =>
        repoMatchesFilter(r, state.filter)
      );
      if (!visible.length) return;
      const idx = visible.findIndex((r) => r.slug === state.activeSlug);
      let next = idx;
      if (e.key === "j") next = idx < 0 ? 0 : Math.min(idx + 1, visible.length - 1);
      if (e.key === "k") next = idx <= 0 ? 0 : idx - 1;
      const target = visible[next];
      if (target && target.slug !== state.activeSlug) {
        location.hash = `#${encodeURIComponent(target.slug)}`;
        selectRepo(target.slug);
      }
      e.preventDefault();
    }
  });

  // -----------------------------------------------------------
  // Data
  // -----------------------------------------------------------
  async function loadRepos() {
    const res = await fetch("/api/repos");
    state.repos = await res.json();
  }

  async function loadStats() {
    const res = await fetch("/api/stats");
    state.stats = await res.json();
  }

  async function loadHealth() {
    const dot = document.getElementById("health-dot");
    try {
      const res = await fetch("/api/health");
      if (!res.ok) throw new Error(`/api/health ${res.status}`);
      const h = await res.json();
      state.health = h;
      if (dot) {
        dot.classList.remove("loading", "err");
        dot.classList.add("ok");
        dot.title = `v${h.version} · ${h.n_repos} repos · ${h.n_envs} envs · started ${h.boot_time}`;
      }
    } catch (e) {
      state.health = null;
      if (dot) {
        dot.classList.remove("loading", "ok");
        dot.classList.add("err");
        dot.title = `health check failed: ${e.message}`;
      }
    }
  }

  async function loadRepoDetail(slug) {
    const res = await fetch(`/api/repos/${encodeURIComponent(slug)}`);
    if (!res.ok) throw new Error(`failed to load ${slug}: ${res.status}`);
    return res.json();
  }

  // -----------------------------------------------------------
  // Top bar
  // -----------------------------------------------------------
  function renderTopbar() {
    const s = state.stats || {};
    topbarStatsEl.innerHTML = "";
    const pieces = [
      ["repos", s.n_repos ?? 0],
      ["modules", s.n_modules ?? 0],
      ["py LOC", fmtInt(s.n_loc ?? 0)],
      ["avg novelty", (s.avg_novelty ?? 0).toFixed(2)],
    ];
    for (const [label, value] of pieces) {
      const el = document.createElement("span");
      el.className = "stat";
      el.innerHTML = `<b>${value}</b> ${label}`;
      topbarStatsEl.appendChild(el);
    }
  }

  // -----------------------------------------------------------
  // Sidebar
  // -----------------------------------------------------------
  function renderSidebar() {
    repoListEl.innerHTML = "";
    const filtered = state.repos.filter((r) => repoMatchesFilter(r, state.filter));
    repoCountEl.textContent = `${filtered.length} / ${state.repos.length}`;
    for (const r of filtered) {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.className = "repo-row" + (r.slug === state.activeSlug ? " active" : "");
      btn.dataset.slug = r.slug;
      btn.innerHTML = `
        <span class="repo-name">${escape(r.name)}</span>
        <span class="pill">${r.n_candidates}</span>
        <span class="repo-sub">${escape(r.selection_mode)} · ${fmtInt(r.n_loc)} LOC · novelty ${
        r.avg_novelty?.toFixed(2) ?? "0.00"
      }</span>
      `;
      btn.addEventListener("click", () => {
        location.hash = `#${encodeURIComponent(r.slug)}`;
        selectRepo(r.slug);
      });
      li.appendChild(btn);
      repoListEl.appendChild(li);
    }
  }

  function repoMatchesFilter(r, f) {
    if (!f) return true;
    if (r.name.toLowerCase().includes(f)) return true;
    if ((r.notes || "").toLowerCase().includes(f)) return true;
    return false;
  }

  // -----------------------------------------------------------
  // Repo detail
  // -----------------------------------------------------------
  async function selectRepo(slug) {
    state.activeSlug = slug;
    renderSidebar();
    contentEl.innerHTML = `<div class="placeholder glass">Loading…</div>`;
    try {
      const record = await loadRepoDetail(slug);
      state.activeRecord = record;
      renderRepoDetail(record);
    } catch (e) {
      contentEl.innerHTML = `<div class="placeholder glass">Failed to load: ${escape(
        e.message
      )}</div>`;
    }
  }

  function renderRepoDetail(record) {
    const candidates = record.candidates.filter((c) => candidateMatchesFilter(c, state.filter));
    candidates.sort((a, b) => b.novelty_score - a.novelty_score);

    const headerHTML = `
      <div class="repo-header glass">
        <div>
          <h2>${escape(record.name)}</h2>
          <p class="repo-url"><a href="${escape(record.url)}" target="_blank" rel="noopener">${escape(
      record.url
    )}</a></p>
          <div class="meta">
            <span class="badge mode-${escape(record.selection_mode)}">mode <b>${escape(
      record.selection_mode
    )}</b></span>
            <span class="badge"><b>${record.candidates.length}</b> modules</span>
            <span class="badge"><b>${fmtInt(record.n_python_files)}</b> py files</span>
            <span class="badge"><b>${fmtInt(record.n_loc)}</b> LOC</span>
            ${
              record.commit_sha
                ? `<span class="badge">sha <b>${escape(record.commit_sha.slice(0, 7))}</b></span>`
                : ""
            }
            <span class="badge">cloned <b>${formatDate(record.cloned_at)}</b></span>
          </div>
          ${record.notes ? `<p class="repo-url" style="margin-top:10px">${escape(record.notes)}</p>` : ""}
        </div>
      </div>
    `;

    const cardsHTML = candidates.length
      ? `<div class="modules-grid">${candidates.map(moduleCard).join("")}</div>`
      : `<div class="placeholder glass">No modules match the current filter.</div>`;

    const traceHTML = record.agent_log && record.agent_log.length
      ? `<div class="trace glass">
           <h3>Agent trace · ${record.agent_log.length} events</h3>
           <div class="trace-list">${record.agent_log
             .slice(0, 60)
             .map(traceItem)
             .join("")}</div>
         </div>`
      : "";

    contentEl.innerHTML = `
      <div class="repo-detail">
        ${headerHTML}
        ${cardsHTML}
        ${traceHTML}
      </div>
    `;

    requestAnimationFrame(() => {
      contentEl.querySelectorAll("pre code").forEach((el) => {
        if (window.hljs) window.hljs.highlightElement(el);
      });
    });
  }

  function moduleCard(c) {
    const nov = Math.round(c.novelty_score * 100);
    const band = noveltyBand(c.novelty_score);
    // We build the absolute path on the server side when possible (via the
    // record's local_path), but the module list endpoint only ships the
    // repo-relative ``file_path``. Copying that is still useful — the agent
    // can prefix it with ``data/clones/<owner>/<repo>/`` themselves.
    return `
      <article class="module-card glass">
        <div class="head">
          <div>
            <div class="title">${escape(c.class_name)}</div>
            <div class="file">
              <span>${escape(c.file_path)} · lines ${c.start_line}-${c.end_line}</span>
              <button
                class="copy-btn"
                data-copy="${escape(c.file_path)}"
                title="Copy ${escape(c.file_path)} to clipboard"
                aria-label="Copy file path"
              >copy</button>
            </div>
          </div>
          <div class="novelty">
            <div class="value">${c.novelty_score.toFixed(2)}</div>
            <div class="novelty-bar"><span class="${band}" style="width:${nov}%"></span></div>
          </div>
        </div>
        <p class="reason">${escape(c.reason)}</p>
        ${
          c.tags && c.tags.length
            ? `<div class="tags">${c.tags
                .map((t) => `<span class="tag">${escape(t)}</span>`)
                .join("")}</div>`
            : ""
        }
        <details>
          <summary>view source · ${c.end_line - c.start_line + 1} LOC</summary>
          <pre><code class="language-python">${escape(c.source_code || "")}</code></pre>
        </details>
      </article>
    `;
  }

  // Single delegated handler — saves us from binding a listener per card.
  document.addEventListener("click", async (ev) => {
    const btn = ev.target.closest("button.copy-btn");
    if (!btn) return;
    const text = btn.dataset.copy || "";
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
      }
      const orig = btn.textContent;
      btn.textContent = "copied";
      btn.classList.add("copied");
      setTimeout(() => {
        btn.textContent = orig;
        btn.classList.remove("copied");
      }, 1200);
    } catch (_e) {
      btn.textContent = "err";
      setTimeout(() => (btn.textContent = "copy"), 1200);
    }
  });

  function traceItem(e) {
    const kind = (e.kind || "").toLowerCase();
    let text = "";
    if (kind === "tool") {
      const inp = JSON.stringify(e.input || {}).slice(0, 120);
      text = `<b>${escape(e.name || "")}</b>(${escape(inp)})`;
    } else {
      text = escape((e.text || "").slice(0, 200));
    }
    return `
      <div class="trace-item">
        <span class="step">${e.step ?? ""}</span>
        <span class="kind ${escape(kind)}">${escape(kind)}</span>
        <span>${text}</span>
      </div>
    `;
  }

  function candidateMatchesFilter(c, f) {
    if (!f) return true;
    if (c.class_name.toLowerCase().includes(f)) return true;
    if (c.file_path.toLowerCase().includes(f)) return true;
    if ((c.reason || "").toLowerCase().includes(f)) return true;
    if ((c.tags || []).some((t) => t.toLowerCase().includes(f))) return true;
    return false;
  }

  // -----------------------------------------------------------
  // Envs / rollouts tab
  // -----------------------------------------------------------
  async function loadAndRenderEnvs() {
    envsContentEl.innerHTML = `<div class="placeholder glass">Loading envs…</div>`;
    try {
      const res = await fetch("/api/envs");
      if (!res.ok) throw new Error(`/api/envs ${res.status}`);
      state.envs = await res.json();
      renderEnvs(state.envs);
    } catch (e) {
      envsContentEl.innerHTML = `<div class="placeholder glass">Failed to load envs: ${escape(
        e.message
      )}</div>`;
    }
  }

  function renderEnvs(envs) {
    const total = envs.length;
    const withReward = envs.filter((e) => e.best_reward !== null && e.best_reward !== undefined);
    const traced = envs.filter((e) => (e.n_traces || 0) > 0);

    // Tally tags across all envs once, then render the top 18 as chips.
    const tagCounts = new Map();
    for (const e of envs) {
      for (const t of e.tags || []) {
        tagCounts.set(t, (tagCounts.get(t) || 0) + 1);
      }
    }
    const topTags = [...tagCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 18);
    const chipsHTML = topTags
      .map(([t, n]) => {
        const active = state.envTagFilter === t ? " active" : "";
        return `<button class="chip${active}" data-tag="${escape(
          t
        )}">${escape(t)} <span class="chip-count">${n}</span></button>`;
      })
      .join("");

    const filtered = state.envTagFilter
      ? envs.filter((e) => (e.tags || []).includes(state.envTagFilter))
      : envs;

    const activeChipNote = state.envTagFilter
      ? `<span class="muted">tag <b>${escape(
          state.envTagFilter
        )}</b> · ${filtered.length} env(s) <button class="chip-clear" id="chip-clear">clear</button></span>`
      : "";

    const summary = `
      <div class="envs-header glass">
        <div>
          <h2>RL envs · ${total}</h2>
          <p class="muted">
            ${traced.length} with at least one rollout · ${withReward.length} with a reward.
            <br />
            Run more:
            <code>python -m kernel_synth.scripts.rollout &lt;env&gt; --mode baseline</code>
            ·
            <code>--mode torch_compile</code>
            ·
            <code>--mode agent</code>
          </p>
          <div class="chips" id="env-chips">${chipsHTML}</div>
          <div class="chips-active">${activeChipNote}</div>
        </div>
      </div>
    `;

    const tableRows = filtered
      .map((e) => {
        const best = fmtReward(e.best_reward);
        const latest = fmtReward(e.latest_reward);
        const traceLink = e.best_trace
          ? `<a href="/api/envs/${encodeURIComponent(
              e.slug
            )}/traces/${encodeURIComponent(e.best_trace)}" target="_blank">${escape(
              e.best_trace
            )}</a>`
          : `<span class="empty-hint" title="Run a rollout to populate this row">
              no rollouts yet —
              <code>python -m kernel_synth.scripts.rollout ${escape(e.slug)}</code>
            </span>`;
        const tags = (e.tags || [])
          .slice(0, 3)
          .map((t) => `<span class="tag">${escape(t)}</span>`)
          .join("");
        const runnable = renderRunnableCell(e);
        return `
          <tr>
            <td class="mono">${escape(e.slug)}</td>
            <td>${escape(e.class_name || "")}</td>
            <td><a href="${escape(e.repo_url || "#")}" target="_blank">${escape(
          e.repo || ""
        )}</a></td>
            <td class="tags-cell">${tags}</td>
            <td>${runnable}</td>
            <td class="right">${e.n_traces || 0}</td>
            <td class="right reward ${rewardClass(e.best_reward)}">${best}</td>
            <td class="right reward ${rewardClass(e.latest_reward)}">${latest}</td>
            <td class="mono">${escape(e.best_mode || e.latest_mode || "")}</td>
            <td>${traceLink}</td>
          </tr>
        `;
      })
      .join("");

    const tableBody = tableRows
      ? tableRows
      : `<tr><td colspan="10" class="empty-row">
           No envs match the current filter. Clear the active chip above to
           see them all.
         </td></tr>`;

    const table = `
      <div class="envs-table-wrap glass">
        <table class="envs-table">
          <thead>
            <tr>
              <th>env</th>
              <th>class</th>
              <th>repo</th>
              <th>tags</th>
              <th>runnable</th>
              <th class="right">traces</th>
              <th class="right">best reward</th>
              <th class="right">latest reward</th>
              <th>best mode</th>
              <th>best trace</th>
            </tr>
          </thead>
          <tbody>${tableBody}</tbody>
        </table>
      </div>
    `;

    envsContentEl.innerHTML = `<div class="repo-detail">${summary}${table}</div>`;

    const chipsEl = document.getElementById("env-chips");
    if (chipsEl) {
      chipsEl.addEventListener("click", (ev) => {
        const btn = ev.target.closest("button[data-tag]");
        if (!btn) return;
        const tag = btn.dataset.tag;
        state.envTagFilter = state.envTagFilter === tag ? null : tag;
        renderEnvs(state.envs);
      });
    }
    const clearBtn = document.getElementById("chip-clear");
    if (clearBtn) {
      clearBtn.addEventListener("click", () => {
        state.envTagFilter = null;
        renderEnvs(state.envs);
      });
    }
  }

  function renderRunnableCell(e) {
    if (e.runnable === true) {
      return `<span class="pill runnable-yes" title="reference imports + inputs.build_module_kwargs() succeeded at build time">ok</span>`;
    }
    if (e.runnable === false) {
      const detail = e.runnable_error ? ` — ${e.runnable_error}` : "";
      return `<span class="pill runnable-no" title="${escape(
        "build-time probe failed" + detail
      )}">err</span>`;
    }
    return `<span class="muted" title="not probed (rebuild envs to populate)">—</span>`;
  }

  function noveltyBand(score) {
    const s = typeof score === "number" ? score : 0;
    if (s >= 0.7) return "band-high";
    if (s >= 0.3) return "band-mid";
    return "band-low";
  }

  function fmtReward(v) {
    if (v === null || v === undefined) return "—";
    return Number(v).toFixed(3);
  }
  function rewardClass(v) {
    if (v === null || v === undefined) return "neutral";
    if (v >= 0.8) return "good";
    if (v >= 0.2) return "ok";
    if (v < 0) return "bad";
    return "neutral";
  }

  // -----------------------------------------------------------
  // Utils
  // -----------------------------------------------------------
  function fmtInt(n) {
    return (n ?? 0).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }
  function formatDate(iso) {
    if (!iso) return "—";
    try {
      const d = new Date(iso);
      return d.toLocaleString();
    } catch (e) {
      return iso;
    }
  }
  function escape(s) {
    return String(s ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  boot();
})();
