(function () {
  "use strict";

  function showToast(message, variant) {
    var container = document.getElementById("toastContainer");
    if (!container || typeof bootstrap === "undefined") {
      window.alert(message);
      return;
    }
    var bg =
      variant === "danger"
        ? "danger"
        : variant === "success"
          ? "success"
          : "dark";
    var el = document.createElement("div");
    el.className = "toast align-items-center text-bg-" + bg + " border-0";
    el.setAttribute("role", "alert");
    el.innerHTML =
      '<div class="d-flex">' +
      '<div class="toast-body">' +
      message +
      "</div>" +
      '<button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>' +
      "</div>";
    container.appendChild(el);
    var t = new bootstrap.Toast(el, { delay: 4500 });
    t.show();
    el.addEventListener("hidden.bs.toast", function () {
      el.remove();
    });
  }

  function parseFloatSafe(v) {
    if (v === undefined || v === null) return NaN;
    var s = String(v).trim().replace(",", ".");
    return parseFloat(s);
  }

  function validatePredictForm(form) {
    var inputs = form.querySelectorAll("input[name]");
    var names = [];
    inputs.forEach(function (inp) {
      if (inp.name) names.push(inp.name);
    });
    for (var i = 0; i < names.length; i++) {
      var name = names[i];
      var inp = form.querySelector('[name="' + name.replace(/"/g, "") + '"]');
      if (!inp) continue;
      var raw = inp.value.trim();
      if (raw === "") {
        showToast("Veuillez remplir tous les champs.", "danger");
        return false;
      }
      var num = parseFloatSafe(raw);
      if (isNaN(num)) {
        showToast("Valeur numérique invalide pour : " + name, "danger");
        return false;
      }
      if (num < 0) {
        showToast("Les valeurs doivent être positives ou nulles : " + name, "danger");
        return false;
      }
      if (name === "Avg_Discount_Rate" && (num < 0 || num > 100)) {
        showToast("Avg_Discount_Rate doit être entre 0 et 100.", "danger");
        return false;
      }
      if (name === "Avg_Delivery_Days" && (num < 0 || num > 30)) {
        showToast("Avg_Delivery_Days doit être entre 0 et 30.", "danger");
        return false;
      }
    }
    return true;
  }

  function setLoader(active) {
    var loader = document.getElementById("pageLoader");
    if (!loader) return;
    if (active) loader.classList.add("active");
    else loader.classList.remove("active");
  }

  function initPredictForm() {
    var form = document.getElementById("predictClientForm");
    if (!form) return;

    form.querySelectorAll("input").forEach(function (el) {
      el.addEventListener("focus", function () {
        el.select();
      });
    });

    form.addEventListener("submit", function (e) {
      if (!validatePredictForm(form)) {
        e.preventDefault();
        setLoader(false);
        return;
      }
      setLoader(true);
      var btn = form.querySelector('button[type="submit"]');
      if (btn) {
        btn.disabled = true;
        if (!btn.hasAttribute("data-orig-html")) {
          btn.setAttribute("data-orig-html", btn.innerHTML);
        }
        btn.innerHTML =
          '<span class="spinner-border spinner-border-sm me-2" role="status"></span>Analyse en cours...';
      }
    });

    var vipBtn = document.getElementById("btnVipExample");
    if (vipBtn) {
      vipBtn.addEventListener("click", function () {
        var map = {
          Total_Sales: "4500",
          Avg_Order_Value: "280",
          Order_Count: "18",
          Total_Profit: "850",
          Avg_Profit_Margin: "0.22",
          Avg_Discount_Rate: "15",
          Avg_Delivery_Days: "3.5",
        };
        Object.keys(map).forEach(function (id) {
          var inp = document.getElementById(id);
          if (inp) {
            inp.value = map[id];
            inp.classList.add("field-flash");
            window.setTimeout(function () {
              inp.classList.remove("field-flash");
            }, 1200);
          }
        });
      });
    }
  }

  function initNavActive() {
    var path = window.location.pathname.replace(/\/$/, "") || "/";
    document.querySelectorAll(".nav-segment-link").forEach(function (a) {
      var p = a.getAttribute("data-path") || "";
      var norm = p.replace(/\/$/, "") || "/";
      var match =
        norm === "/"
          ? path === "/" || path === ""
          : path === norm;
      if (match) {
        a.classList.add("active");
      }
    });
  }

  function initTooltips() {
    if (typeof bootstrap === "undefined") return;
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(function (el) {
      new bootstrap.Tooltip(el);
    });
  }

  function initScrollAnimations() {
    var els = document.querySelectorAll(".animate-fade-in-up");
    if (!els.length || !("IntersectionObserver" in window)) {
      els.forEach(function (el) {
        el.classList.add("visible");
      });
      return;
    }
    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (en) {
          if (en.isIntersecting) {
            en.target.classList.add("visible");
            io.unobserve(en.target);
          }
        });
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0.05 }
    );
    els.forEach(function (el) {
      io.observe(el);
    });
  }

  function initProgressBars() {
    document.querySelectorAll(".progress-bar-animate").forEach(function (bar, idx) {
      var pct = bar.getAttribute("data-pct") || "100";
      bar.style.setProperty("--target-width", pct + "%");
      window.setTimeout(function () {
        bar.classList.add("in-view");
      }, 120 + idx * 80);
    });
  }

  function sortClusterKeys(keys) {
    return keys.slice().sort(function (a, b) {
      return parseInt(a, 10) - parseInt(b, 10);
    });
  }

  function initDashboardCharts() {
    var el = document.getElementById("dash-profiles-json");
    var doughCanvas = document.getElementById("dashDoughnut");
    var radarCanvas = document.getElementById("dashRadar");
    if (!el || !doughCanvas || !radarCanvas || typeof Chart === "undefined") return;

    var profiles;
    try {
      profiles = JSON.parse(el.textContent || "{}");
    } catch (err) {
      return;
    }

    var keys = sortClusterKeys(Object.keys(profiles));
    var counts = keys.map(function (k) {
      return profiles[k].count || 0;
    });
    var labelMap = {
      0: "Cluster 0 — Moyen actif",
      1: "Cluster 1 — VIP",
      2: "Cluster 2 — Intermédiaire",
      3: "Cluster 3 — Atypique",
    };
    var doughLabels = keys.map(function (k) {
      var n = parseInt(k, 10);
      return labelMap[n] || "Cluster " + k;
    });
    var colors = ["#2DC653", "#1E90FF", "#F4A261", "#7B2D8B"];

    new Chart(doughCanvas.getContext("2d"), {
      type: "doughnut",
      data: {
        labels: doughLabels,
        datasets: [
          {
            data: counts,
            backgroundColor: colors.slice(0, keys.length),
            borderWidth: 2,
            borderColor: "#fff",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "bottom" },
        },
      },
    });

    var firstKey = keys[0];
    if (!firstKey || !profiles[firstKey].means) return;

    var meanKeys = Object.keys(profiles[firstKey].means);
    var radarAxes = meanKeys.slice(0, 6);

    var rawMatrix = keys.map(function (cid) {
      return radarAxes.map(function (axis) {
        var m = profiles[cid].means || {};
        return typeof m[axis] === "number" ? m[axis] : parseFloat(m[axis]) || 0;
      });
    });

    var normMatrix = rawMatrix.map(function (row) {
      return row.map(function (_, j) {
        var col = rawMatrix.map(function (r) {
          return r[j];
        });
        var mn = Math.min.apply(null, col);
        var mx = Math.max.apply(null, col);
        var span = mx - mn || 1;
        return (row[j] - mn) / span;
      });
    });

    var datasets = keys.map(function (cid, i) {
      return {
        label: "Cluster " + cid,
        data: normMatrix[i],
        borderColor: colors[i % colors.length],
        backgroundColor: colors[i % colors.length] + "33",
        borderWidth: 2,
        pointRadius: 3,
      };
    });

    new Chart(radarCanvas.getContext("2d"), {
      type: "radar",
      data: {
        labels: radarAxes,
        datasets: datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            min: 0,
            max: 1,
            ticks: { stepSize: 0.25, display: false },
            grid: { color: "rgba(13,27,42,0.08)" },
            angleLines: { color: "rgba(13,27,42,0.1)" },
            pointLabels: { font: { size: 10 } },
          },
        },
        plugins: {
          legend: { position: "bottom" },
        },
      },
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    initNavActive();
    initTooltips();
    initPredictForm();
    initScrollAnimations();
    initProgressBars();
    initDashboardCharts();
  });
})();
