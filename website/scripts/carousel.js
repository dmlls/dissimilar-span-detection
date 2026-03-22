[...document.getElementsByClassName("carousel-btn")]?.forEach((btn) => {
  btn.addEventListener("click", (e) => {
    e.preventDefault();
    const scrollTop =
      document.documentElement.scrollTop || document.body.scrollTop;
    location.hash = btn.getAttribute("href");
    document.documentElement.scrollTop = scrollTop;
  });
});
