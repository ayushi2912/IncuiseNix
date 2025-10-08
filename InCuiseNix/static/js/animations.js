// Simple script for the fade-in-on-scroll effect
document.addEventListener("DOMContentLoaded", function() {
  const sections = document.querySelectorAll('.fade-in-section');

  const sectionObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  sections.forEach(section => {
    sectionObserver.observe(section);
  });
});