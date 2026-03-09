<script setup lang="ts">
import { computed, onMounted, ref } from "vue";

const plannedFeatures = [
  "OCR-first parsing for menu images and PDFs",
  "Category prediction for each menu item",
  "Structured extraction for name, price, size, and allergens",
  "Human-in-the-loop review and export",
];

type ApiStatus = {
  service: string;
  status: string;
  version: string;
};

const apiState = ref<"idle" | "loading" | "online" | "error">("idle");
const apiVersion = ref("unknown");
const apiError = ref("");

const apiHeadline = computed(() => {
  if (apiState.value === "loading") {
    return "Checking backend connectivity";
  }

  if (apiState.value === "online") {
    return "Frontend and backend are communicating";
  }

  if (apiState.value === "error") {
    return "Backend is unreachable";
  }

  return "Backend check has not started yet";
});

async function checkApiStatus(): Promise<void> {
  apiState.value = "loading";
  apiError.value = "";

  try {
    const response = await fetch("/api/status", {
      headers: {
        Accept: "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const payload = (await response.json()) as ApiStatus;
    apiState.value = payload.status === "ok" ? "online" : "error";
    apiVersion.value = payload.version;
  } catch (error) {
    apiState.value = "error";
    apiError.value = error instanceof Error ? error.message : "Unknown error";
  }
}

onMounted(() => {
  void checkApiStatus();
});
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
          <strong>{{ apiHeadline }}</strong>
          <p>
            Status:
            <span class="status-pill" :class="`state-${apiState}`">
              {{ apiState }}
            </span>
          </p>
          <p>API version: {{ apiVersion }}</p>
          <p v-if="apiError" class="error-copy">Last error: {{ apiError }}</p>
          <button class="refresh-button" type="button" @click="checkApiStatus">
            Check API again
          </button>
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
