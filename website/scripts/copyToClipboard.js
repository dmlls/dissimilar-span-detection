const copyCitationBtn = document.getElementById("copy-citation");
const citation = document.getElementById("citation");
if (copyCitationBtn != null && citation != null) {
  copyCitationBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    await navigator.clipboard.writeText(citation.textContent);
    const tooltip = e.target.closest("div.d-tooltip");
    tooltip?.setAttribute("data-tip", "Copied");
    tooltip?.classList.add("d-tooltip-success");
    copyCitationBtn.classList.add("copied");
    const svgs = [...e.target.closest("#copy-citation")?.children];
    svgs.forEach((e) => {
      e.classList.toggle("active");
      e.classList.toggle("hidden");
    });
    setTimeout(() => {
      tooltip?.setAttribute("data-tip", "Copy");
      tooltip?.classList.remove("d-tooltip-success");
      copyCitationBtn.classList.remove("copied");
      svgs.forEach((e) => {
        e.classList.toggle("active");
        e.classList.toggle("hidden");
      });
    }, 1000);
  });
}
