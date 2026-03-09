<script setup lang="ts">
const plannedFeatures = [
  "OCR-first parsing for menu images and PDFs",
  "Category prediction for each menu item",
  "Structured extraction for name, price, size, and allergens",
  "Human-in-the-loop review and export",
];
</script>

<template>
  <main class="page-shell">
    <section class="hero-card">
      <p class="eyebrow">NLP Course Project</p>
      <h1>Menu Intelligence</h1>
      <p class="lead">
        A web application that turns raw menu text or menu documents into a structured
        menu ready for review, export, and downstream automation.
      </p>

      <div class="status-grid">
        <article class="status-card">
          <span class="status-label">Frontend</span>
          <strong>Base scaffold ready</strong>
          <p>Vue 3, Vite, and TypeScript are wired and ready for feature work.</p>
        </article>
        <article class="status-card">
          <span class="status-label">Backend</span>
          <strong>API base ready</strong>
          <p>FastAPI endpoints provide health and version checks for deployment flow.</p>
        </article>
      </div>
    </section>

    <section class="roadmap-card">
      <div>
        <p class="eyebrow">Initial roadmap</p>
        <h2>Next implementation milestones</h2>
      </div>

      <ul class="feature-list">
        <li v-for="feature in plannedFeatures" :key="feature">
          {{ feature }}
        </li>
      </ul>
    </section>
  </main>
</template>

