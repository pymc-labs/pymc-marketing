// rtd-stale-banner.js
// Persistent "stale docs" banner injected via RTD Custom Script addon.
// Reads addons data from window.ReadTheDocsEventData (already available when
// this script runs) and renders a fixed top banner on non-stable versions.
// See: https://github.com/readthedocs/addons/issues/413
//      https://github.com/pymc-labs/pymc-marketing/issues/942

(function () {
  "use strict";

  var STORAGE_KEY_PREFIX = "rtd-stale-banner-dismissed";
  var DISMISS_DAYS = 30;

  function getDismissKey(data) {
    var slug =
      data && data.versions && data.versions.current
        ? data.versions.current.slug
        : "unknown";
    return STORAGE_KEY_PREFIX + "-" + slug;
  }

  function isDismissed(key) {
    try {
      var raw = localStorage.getItem(key);
      if (!raw) return false;
      var ts = parseInt(raw, 10);
      if (isNaN(ts)) return false;
      var ageMs = Date.now() - ts;
      return ageMs < DISMISS_DAYS * 24 * 60 * 60 * 1000;
    } catch (_) {
      return false;
    }
  }

  function dismiss(key) {
    try {
      localStorage.setItem(key, String(Date.now()));
    } catch (_) {
      /* quota or private browsing — ignore */
    }
  }

  function buildStableUrl(pathname) {
    // Rewrite /<lang>/<version>/... → /<lang>/stable/...
    return pathname.replace(/^\/(en|es)\/[^/]+\//, "/$1/stable/");
  }

  function getVersionType(data) {
    var slug = data.versions.current.slug;
    var type = data.versions.current.type;
    var defaultVersion = data.projects.current.default_version;

    if (slug === defaultVersion) return "stable";
    if (slug === "latest") return "latest";
    if (type === "tag") return "old";
    return "other";
  }

  function render(data) {
    var version = data.versions.current.slug;
    var versionType = getVersionType(data);

    if (versionType === "stable") return;

    var key = getDismissKey(data);
    if (isDismissed(key)) return;

    var isDev = versionType === "latest";
    var banner = document.createElement("div");
    banner.id = "rtd-stale-banner";
    banner.setAttribute("role", "alert");

    // Color scheme: blue for dev (info), yellow for old (warning)
    var bg = isDev ? "#d1ecf1" : "#fef3cd";
    var border = isDev ? "#bee5eb" : "#ffc107";
    var textColor = isDev ? "#0c5460" : "#1a1a2e";

    Object.assign(banner.style, {
      position: "fixed",
      top: "0",
      left: "0",
      right: "0",
      zIndex: "9999",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      gap: "0.75em",
      padding: "0.6em 1em",
      fontFamily:
        '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      fontSize: "14px",
      lineHeight: "1.4",
      color: textColor,
      background: bg,
      borderBottom: "1px solid " + border,
      boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
      boxSizing: "border-box",
      textAlign: "center",
      flexWrap: "wrap",
    });

    var stableUrl = buildStableUrl(window.location.pathname);

    var msg = document.createElement("span");
    if (isDev) {
      msg.textContent =
        "This is the development version of the documentation. Features shown here may not be available in the latest stable release. ";
    } else {
      msg.textContent =
        "You are viewing docs for version " +
        version +
        ", which may be outdated. ";
    }

    var link = document.createElement("a");
    link.href = stableUrl;
    link.textContent = "Go to stable docs";
    Object.assign(link.style, {
      color: "#0d6efd",
      fontWeight: "600",
      textDecoration: "underline",
    });

    var closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.setAttribute("aria-label", "Dismiss");
    closeBtn.textContent = "\u00d7"; // ×
    Object.assign(closeBtn.style, {
      background: "none",
      border: "none",
      fontSize: "18px",
      fontWeight: "700",
      lineHeight: "1",
      color: "#666",
      cursor: "pointer",
      padding: "0 0.25em",
      marginLeft: "0.5em",
    });

    closeBtn.addEventListener("click", function () {
      dismiss(key);
      banner.remove();
      // Shift page content back up
      document.body.style.marginTop = "";
    });

    banner.appendChild(msg);
    banner.appendChild(link);
    banner.appendChild(closeBtn);
    document.body.prepend(banner);

    // Shift page content down to avoid overlap
    document.body.style.marginTop = banner.offsetHeight + "px";
  }

  function init() {
    if (!window.ReadTheDocsEventData) return;

    var data;
    try {
      data = window.ReadTheDocsEventData.data(true);
    } catch (_) {
      return;
    }

    if (!data || !data.versions || !data.versions.current) return;

    render(data);
  }

  // ReadTheDocsEventData is available immediately (custom scripts load after
  // the addons data event has fired).
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
