(() => {
  const body = document.body;
  const toggle = document.querySelector("[data-nav-toggle]");
  const panel = document.querySelector("[data-nav-panel]");

  const closeNav = () => {
    body.classList.remove("nav-open");
    if (toggle) {
      toggle.setAttribute("aria-expanded", "false");
    }
  };

  if (toggle && panel) {
    toggle.addEventListener("click", () => {
      const isOpen = body.classList.toggle("nav-open");
      toggle.setAttribute("aria-expanded", String(isOpen));
    });

    panel.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", closeNav);
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        closeNav();
      }
    });

    window.matchMedia("(min-width: 961px)").addEventListener("change", (event) => {
      if (event.matches) {
        closeNav();
      }
    });
  }

  // Desktop dropdowns
  document.querySelectorAll(".nav__dropdown-toggle").forEach((button) => {
    const dropdown = button.closest(".nav__dropdown");
    const menu = dropdown.querySelector(".nav__dropdown-menu");

    button.addEventListener("click", (e) => {
      e.stopPropagation();
      const isActive = dropdown.classList.toggle("active");
      button.setAttribute("aria-expanded", String(isActive));
    });

    // Close dropdown when clicking outside
    document.addEventListener("click", (e) => {
      if (!dropdown.contains(e.target)) {
        dropdown.classList.remove("active");
        button.setAttribute("aria-expanded", "false");
      }
    });

    // Close dropdown on escape
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && dropdown.classList.contains("active")) {
        dropdown.classList.remove("active");
        button.setAttribute("aria-expanded", "false");
      }
    });
  });

  // Mobile dropdowns
  document.querySelectorAll(".nav__mobile-dropdown-toggle").forEach((button) => {
    const dropdown = button.closest(".nav__mobile-dropdown");

    button.addEventListener("click", () => {
      const isActive = dropdown.classList.toggle("active");
      button.setAttribute("aria-expanded", String(isActive));
    });
  });

  const yearEl = document.getElementById("year");
  if (yearEl) {
    yearEl.textContent = new Date().getFullYear().toString();
  }

  // Carousel functionality
  const carouselTrack = document.getElementById("carouselTrack");
  const carouselPrev = document.getElementById("carouselPrev");
  const carouselNext = document.getElementById("carouselNext");
  const carouselDots = document.querySelectorAll(".carousel-dot");

  if (carouselTrack && carouselDots.length > 0) {
    let currentSlide = 0;
    const totalSlides = carouselDots.length;
    let autoPlayInterval = null;

    function updateCarousel() {
      const translateX = -currentSlide * 100;
      carouselTrack.style.transform = `translate3d(${translateX}%, 0, 0)`;
      
      carouselDots.forEach((dot, index) => {
        if (index === currentSlide) {
          dot.classList.add("active");
        } else {
          dot.classList.remove("active");
        }
      });

      // Update text content below carousel
      const activeSlide = carouselTrack.querySelector(`[data-slide-index="${currentSlide}"]`);
      const titleEl = document.getElementById("carouselTitle");
      const descriptionEl = document.getElementById("carouselDescription");
      
      if (activeSlide && titleEl && descriptionEl) {
        const title = activeSlide.getAttribute("data-title");
        const description = activeSlide.getAttribute("data-description");
        
        if (title) {
          titleEl.style.opacity = "0";
          setTimeout(() => {
            titleEl.textContent = title;
            titleEl.style.opacity = "1";
          }, 150);
        }
        
        if (description) {
          descriptionEl.style.opacity = "0";
          setTimeout(() => {
            descriptionEl.textContent = description;
            descriptionEl.style.opacity = "1";
          }, 150);
        }
      }
    }

    function nextSlide() {
      currentSlide = (currentSlide + 1) % totalSlides;
      updateCarousel();
    }

    function prevSlide() {
      currentSlide = (currentSlide - 1 + totalSlides) % totalSlides;
      updateCarousel();
    }

    function goToSlide(index) {
      currentSlide = index;
      updateCarousel();
    }

    function startAutoPlay() {
      stopAutoPlay();
      autoPlayInterval = setInterval(nextSlide, 5000);
    }

    function stopAutoPlay() {
      if (autoPlayInterval) {
        clearInterval(autoPlayInterval);
        autoPlayInterval = null;
      }
    }

    updateCarousel();
    startAutoPlay();
    
    if (carouselNext) {
      carouselNext.addEventListener("click", (e) => {
        e.preventDefault();
        nextSlide();
        stopAutoPlay();
        startAutoPlay();
      });
    }
    
    if (carouselPrev) {
      carouselPrev.addEventListener("click", (e) => {
        e.preventDefault();
        prevSlide();
        stopAutoPlay();
        startAutoPlay();
      });
    }
    
    carouselDots.forEach((dot, index) => {
      dot.addEventListener("click", (e) => {
        e.preventDefault();
        goToSlide(index);
        stopAutoPlay();
        startAutoPlay();
      });
    });
    
    const carouselContainer = document.querySelector(".carousel-container");
    if (carouselContainer) {
      carouselContainer.addEventListener("mouseenter", stopAutoPlay);
      carouselContainer.addEventListener("mouseleave", startAutoPlay);
    }
    
    const carousel = document.querySelector(".hero-carousel");
    if (carousel) {
      const handleKeydown = (e) => {
        if (document.activeElement === document.body || carousel.contains(document.activeElement)) {
          if (e.key === "ArrowLeft") {
            e.preventDefault();
            prevSlide();
            stopAutoPlay();
            startAutoPlay();
          } else if (e.key === "ArrowRight") {
            e.preventDefault();
            nextSlide();
            stopAutoPlay();
            startAutoPlay();
          }
        }
      };
      
      document.addEventListener("keydown", handleKeydown);
    }
  }

  // Media carousels in sections
  document.querySelectorAll("[data-carousel]").forEach((carousel) => {
    const track = carousel.querySelector("[data-carousel-track]");
    const slides = carousel.querySelectorAll("[data-carousel-slide]");
    const prev = carousel.querySelector("[data-carousel-prev]");
    const next = carousel.querySelector("[data-carousel-next]");
    const dots = carousel.querySelectorAll("[data-carousel-dot]");

    if (!track || slides.length === 0) {
      return;
    }

    if (slides.length <= 1) {
      if (prev) {
        prev.hidden = true;
      }
      if (next) {
        next.hidden = true;
      }
      const dotsContainer = carousel.querySelector("[data-carousel-dots]");
      if (dotsContainer) {
        dotsContainer.hidden = true;
      }
      return;
    }

    let currentIndex = 0;

    const update = () => {
      track.style.transform = `translate3d(-${currentIndex * 100}%, 0, 0)`;
      dots.forEach((dot, index) => {
        const isActive = index === currentIndex;
        dot.classList.toggle("is-active", isActive);
        dot.setAttribute("aria-selected", String(isActive));
      });
    };

    const goTo = (index) => {
      currentIndex = (index + slides.length) % slides.length;
      update();
    };

    if (prev) {
      prev.addEventListener("click", () => goTo(currentIndex - 1));
    }

    if (next) {
      next.addEventListener("click", () => goTo(currentIndex + 1));
    }
    dots.forEach((dot) => {
      const targetIndex = Number(dot.dataset.carouselDot) || 0;
      dot.addEventListener("click", () => goTo(targetIndex));
    });

    update();
  });
})();

