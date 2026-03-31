const tabButtons = document.querySelectorAll(".tab-button");
const sections = document.querySelectorAll(".chapter, .hero");
const navLinks = document.querySelectorAll("[data-nav]");
const revealEls = document.querySelectorAll(".reveal");

tabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const group = button.dataset.group;
    const target = button.dataset.target;

    document
      .querySelectorAll(`.tab-button[data-group="${group}"]`)
      .forEach((item) => item.classList.remove("active"));

    document
      .querySelectorAll(`.tab-panel[data-group="${group}"]`)
      .forEach((panel) => panel.classList.remove("active"));

    button.classList.add("active");
    document.getElementById(target)?.classList.add("active");
  });
});

const navObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) {
        return;
      }

      const id = entry.target.id;
      navLinks.forEach((link) => {
        link.classList.toggle("active", link.getAttribute("href") === `#${id}`);
      });
    });
  },
  {
    threshold: 0.35,
    rootMargin: "-10% 0px -40% 0px",
  }
);

sections.forEach((section) => {
  if (section.id) {
    navObserver.observe(section);
  }
});

const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
      }
    });
  },
  {
    threshold: 0.15,
  }
);

revealEls.forEach((el) => revealObserver.observe(el));
