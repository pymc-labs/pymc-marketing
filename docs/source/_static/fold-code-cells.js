// Fold notebook code cells by default (#2926).
//
// myst-nb renders every notebook code cell as
//   <div class="cell"><div class="cell_input">...</div><div class="cell_output">...</div></div>
// This script wraps each cell_input in a <details> element that starts
// collapsed, so readers see the narrative and the outputs first and can
// expand the code they care about. A per-page "Show all code" toggle is
// added above the first folded cell.

(function () {
  "use strict";

  var LABELS = {
    en: { show: "Show code", hide: "Hide code", showAll: "Show all code", hideAll: "Hide all code" },
    es: { show: "Mostrar código", hide: "Ocultar código", showAll: "Mostrar todo el código", hideAll: "Ocultar todo el código" },
  };

  function labels() {
    var lang = (document.documentElement.lang || "en").split("-")[0];
    return LABELS[lang] || LABELS.en;
  }

  function foldCell(input, t) {
    var details = document.createElement("details");
    details.className = "code-fold";

    var summary = document.createElement("summary");
    summary.className = "code-fold__summary";
    summary.dataset.show = t.show;
    summary.dataset.hide = t.hide;
    summary.textContent = t.show;
    details.appendChild(summary);

    details.addEventListener("toggle", function () {
      summary.textContent = details.open ? summary.dataset.hide : summary.dataset.show;
    });

    input.parentNode.insertBefore(details, input);
    details.appendChild(input);
    return details;
  }

  function addGlobalToggle(firstCell, folds, t) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "code-fold__toggle-all";
    button.textContent = t.showAll;

    button.addEventListener("click", function () {
      var expand = button.textContent === t.showAll;
      folds.forEach(function (details) {
        details.open = expand;
      });
      button.textContent = expand ? t.hideAll : t.showAll;
    });

    firstCell.parentNode.insertBefore(button, firstCell);
  }

  document.addEventListener("DOMContentLoaded", function () {
    var t = labels();
    var inputs = document.querySelectorAll("div.cell > div.cell_input");
    if (inputs.length === 0) {
      return;
    }

    var folds = [];
    inputs.forEach(function (input) {
      folds.push(foldCell(input, t));
    });

    addGlobalToggle(folds[0].closest("div.cell"), folds, t);
  });
})();
