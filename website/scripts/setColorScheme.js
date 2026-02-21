const themes = {
  lightTheme: "winter",
  darkTheme: "night",
};

let prefersDark =
  window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)");
prefersDark = prefersDark && prefersDark?.matches;
const storedPreference = localStorage.getItem("prefersDark");
if (storedPreference != null) {
  prefersDark = storedPreference === "true";
}
const themeToggle = document.getElementById("theme-toggle");
themeToggle.checked = !prefersDark;
let currentTheme = prefersDark ? themes.darkTheme : themes.lightTheme;
themeToggle.value = currentTheme;
document.documentElement.setAttribute("data-theme", currentTheme);

themeToggle.addEventListener("click", () => {
  currentTheme =
    currentTheme === themes.lightTheme ? themes.darkTheme : themes.lightTheme;
  prefersDark = !prefersDark;
  localStorage.setItem("prefersDark", prefersDark.toString());
  themeToggle.checked = !prefersDark;
  themeToggle.value = currentTheme;
  document.documentElement.setAttribute("data-theme", currentTheme);
});
