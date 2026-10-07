// Publication resources.
const links = {
  paper: "https://arxiv.org/pdf/2610.07250",
  arxiv: "https://arxiv.org/abs/2610.07250",
};

for (const [name, url] of Object.entries(links)) {
  if (!url) continue;
  const placeholder = document.querySelector(`[data-resource="${name}"]`);
  const link = document.createElement("a");
  link.className = placeholder.className;
  link.append(...placeholder.childNodes);
  link.href = url;
  placeholder.replaceWith(link);
}

const examples = [
  { label: "Car & mouse", query: "a photo of a car and a computer mouse" },
  { label: "Couch & cup", query: "a photo of a couch below a cup" },
  { label: "Lion & glass cows", query: "a lion under four glass cows" },
  { label: "Irish symbol", query: "Show a plant that is a symbol of good fortune in Irish culture, and is known for its three-lobed leaves" },
];
const modes = ["Direct generation", "Skill-free harness", "Evolved skills"];
let selectedExample = 0;
let selectedMode = 0;
const baseImage = document.querySelector("#base-image");
const updatedImage = document.querySelector("#updated-image");

function updateGallery() {
  const example = examples[selectedExample];
  for (const [image, column, generator] of [
    [baseImage, selectedMode + 1, "Base generator"],
    [updatedImage, selectedMode + 4, "D-OPCD generator"],
  ]) {
    image.src = `assets/example-${selectedExample + 1}-${column}.webp`;
    image.alt = `${generator}, ${modes[selectedMode].toLowerCase()}: ${example.query}`;
    image.parentElement.setAttribute("aria-label", `Enlarge ${image.alt}`);
  }
  document.querySelector("#query-text").textContent = example.query;
  document.querySelector("#base-mode").textContent = selectedMode === 2 ? "Evolved skills" : modes[selectedMode];
  document.querySelector("#updated-mode").textContent = selectedMode === 2 ? "Newly evolved skills" : modes[selectedMode];
  document.querySelector("#gallery-status").textContent = `Example ${selectedExample + 1} of 4. ${modes[selectedMode]}. ${example.query}`;
  document.querySelectorAll("[data-example]").forEach((button, index) => {
    button.setAttribute("aria-pressed", String(index === selectedExample));
  });
}

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    selectedExample = Number(button.dataset.example);
    updateGallery();
  });
});

function wireTabs(tablist, onSelect) {
  const tabs = [...tablist.querySelectorAll('[role="tab"]')];
  function select(tab) {
    tabs.forEach((item) => {
      const active = item === tab;
      item.setAttribute("aria-selected", String(active));
      item.tabIndex = active ? 0 : -1;
    });
    onSelect(tab, tabs.indexOf(tab));
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => select(tab));
    tab.addEventListener("keydown", (event) => {
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      if (next === undefined) return;
      event.preventDefault();
      select(tabs[next]);
      tabs[next].focus();
    });
  });
}

wireTabs(document.querySelector("#mode-tabs"), (tab, index) => {
  selectedMode = index;
  document.querySelector("#gallery-panel").setAttribute("aria-labelledby", tab.id);
  updateGallery();
});
wireTabs(document.querySelector("#result-tabs"), (tab) => {
  document.querySelectorAll("[data-result-panel]").forEach((panel) => {
    panel.hidden = panel.id !== tab.getAttribute("aria-controls");
  });
});

const lightbox = document.querySelector("#lightbox");
let imageOpener;
document.querySelectorAll("[data-enlarge]").forEach((button) => {
  button.addEventListener("click", () => {
    const image = button.querySelector("img");
    const enlarged = document.querySelector("#enlarged-image");
    enlarged.src = image.src;
    enlarged.alt = image.alt;
    document.querySelector("#lightbox-caption").textContent = image.alt;
    imageOpener = button;
    lightbox.showModal();
  });
});
document.querySelector("#close-lightbox").addEventListener("click", () => lightbox.close());
lightbox.addEventListener("click", (event) => {
  if (event.target !== lightbox) return;
  const box = lightbox.getBoundingClientRect();
  if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) lightbox.close();
});
lightbox.addEventListener("close", () => imageOpener?.focus());

document.querySelector("#copy-citation").addEventListener("click", async () => {
  const button = document.querySelector("#copy-citation");
  const status = document.querySelector("#copy-status");
  const citation = document.querySelector("#bibtex").textContent;
  try {
    await navigator.clipboard.writeText(citation);
    button.textContent = "Copied";
    status.textContent = "BibTeX copied to clipboard.";
    setTimeout(() => { button.textContent = "Copy BibTeX"; }, 2500);
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.querySelector("#bibtex"));
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = "Select and copy the highlighted BibTeX.";
  }
});
