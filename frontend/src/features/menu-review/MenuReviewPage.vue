<script setup lang="ts">
import { computed, ref } from "vue";
import {
  ApiError,
  parseMenu,
  type IssueLevel,
  type MenuItemV1,
  type ParseMenuResponseV1,
} from "../../api/menu";
import {
  CATEGORY_LABELS,
  CURRENCY_OPTIONS,
  DEFAULT_PARSE_REQUEST,
  KIND_OPTIONS,
  SIZE_UNIT_OPTIONS,
} from "./schema";

const text = ref("");
const loading = ref(false);
const errorMessage = ref("");
const result = ref<ParseMenuResponseV1 | null>(null);

const canSubmit = computed(() => text.value.trim().length > 0 && !loading.value);
const items = computed(() => result.value?.items ?? []);

function cloneResponse(value: ParseMenuResponseV1): ParseMenuResponseV1 {
  return JSON.parse(JSON.stringify(value)) as ParseMenuResponseV1;
}

function formatConfidence(value?: number | null): string {
  if (typeof value !== "number") {
    return "-";
  }

  return `${Math.round(value * 100)}%`;
}

function issueClass(level: IssueLevel): string {
  return `ds-issue ds-issue--${level}`;
}

function toNullable(value: string): string | null {
  const next = value.trim();
  return next.length > 0 ? next : null;
}

async function submit(): Promise<void> {
  if (!canSubmit.value) {
    return;
  }

  loading.value = true;
  errorMessage.value = "";

  try {
    const response = await parseMenu({
      schema_version: DEFAULT_PARSE_REQUEST.schema_version,
      text: text.value,
      lang: DEFAULT_PARSE_REQUEST.lang,
      currency_hint: DEFAULT_PARSE_REQUEST.currency_hint,
      category_labels: [...CATEGORY_LABELS],
    });

    result.value = cloneResponse(response);
  } catch (error) {
    result.value = null;
    errorMessage.value =
      error instanceof ApiError ? error.message : "Unexpected error.";
  } finally {
    loading.value = false;
  }
}

function onCategoryChange(item: MenuItemV1, event: Event): void {
  const target = event.target as HTMLSelectElement;
  item.category.label = target.value || null;
}

function onNameInput(item: MenuItemV1, event: Event): void {
  const target = event.target as HTMLInputElement;
  item.fields.name = toNullable(target.value);
}

function onDescriptionInput(item: MenuItemV1, event: Event): void {
  const target = event.target as HTMLTextAreaElement;
  item.fields.description = toNullable(target.value);
}

function addPrice(item: MenuItemV1): void {
  if (item.fields.prices[0]) {
    return;
  }

  item.fields.prices = [
    { value: 0, currency: DEFAULT_PARSE_REQUEST.currency_hint },
  ];
}

function removePrice(item: MenuItemV1): void {
  item.fields.prices = [];
}

function onPriceValueInput(item: MenuItemV1, event: Event): void {
  const target = event.target as HTMLInputElement;
  const raw = target.value.trim();

  if (!raw) {
    item.fields.prices = [];
    return;
  }

  const value = Number(raw.replace(",", "."));

  if (Number.isNaN(value)) {
    return;
  }

  const current = item.fields.prices[0];

  item.fields.prices = [
    {
      value,
      currency: current?.currency ?? DEFAULT_PARSE_REQUEST.currency_hint,
      raw: current?.raw,
    },
  ];
}

function onPriceCurrencyChange(item: MenuItemV1, event: Event): void {
  const current = item.fields.prices[0];

  if (!current) {
    return;
  }

  const target = event.target as HTMLSelectElement;
  current.currency = target.value;
}

function addSize(item: MenuItemV1): void {
  if (item.fields.sizes[0]) {
    return;
  }

  item.fields.sizes = [{ value: 0, unit: SIZE_UNIT_OPTIONS[0] }];
}

function removeSize(item: MenuItemV1): void {
  item.fields.sizes = [];
}

function onSizeValueInput(item: MenuItemV1, event: Event): void {
  const target = event.target as HTMLInputElement;
  const raw = target.value.trim();

  if (!raw) {
    item.fields.sizes = [];
    return;
  }

  const value = Number(raw.replace(",", "."));

  if (Number.isNaN(value)) {
    return;
  }

  const current = item.fields.sizes[0];

  item.fields.sizes = [
    {
      value,
      unit: current?.unit ?? SIZE_UNIT_OPTIONS[0],
      raw: current?.raw,
    },
  ];
}

function onSizeUnitChange(item: MenuItemV1, event: Event): void {
  const current = item.fields.sizes[0];

  if (!current) {
    return;
  }

  const target = event.target as HTMLSelectElement;
  current.unit = target.value;
}
</script>

<template>
  <main class="page">
    <section class="hero panel ds-card">
      <div class="hero__copy">
        <p class="ds-label ds-eyebrow">Slice 1 Review Flow</p>
        <h1 class="ds-title ds-title--hero">
          Menu review starts with raw text and a fixed contract.
        </h1>
        <p class="ds-lead">
          Paste menu text, submit it to the backend parser, and review the
          first normalized fields without introducing frontend-side model logic.
        </p>
      </div>
      <div class="hero__meta">
        <span class="ds-pill">POST /api/v1/menu/parse</span>
        <span class="ds-pill">same-origin /api</span>
        <span class="ds-pill">manual review enabled</span>
      </div>
    </section>

    <section class="panel ds-card">
      <div class="panel__header">
        <div>
          <p class="ds-label ds-eyebrow">Input</p>
          <h2 class="ds-title ds-title--section">Paste raw menu text</h2>
        </div>
        <button
          data-testid="parse-button"
          class="ds-button"
          :disabled="!canSubmit"
          @click="submit"
        >
          {{ loading ? "Parsing..." : "Parse menu" }}
        </button>
      </div>

      <label class="ds-field ds-field--stacked">
        <span class="ds-label">Raw menu text</span>
        <textarea
          v-model="text"
          class="ds-control"
          data-testid="menu-text"
          rows="12"
          placeholder="САЛАТЫ&#10;Цезарь с курицей 250 г - 390 ₽&#10;Греческий - 350 ₽"
        />
      </label>

      <p v-if="errorMessage" class="error">{{ errorMessage }}</p>
    </section>

    <section v-if="result" class="panel panel--compact ds-card">
      <div class="summary">
        <div>
          <span class="ds-label">Request</span>
          <span>{{ result.request_id }}</span>
        </div>
        <div>
          <span class="ds-label">Items</span>
          <span>{{ result.items.length }}</span>
        </div>
        <div>
          <span class="ds-label">Lang</span>
          <span>{{ result.meta.lang }}</span>
        </div>
        <div>
          <span class="ds-label">Currency</span>
          <span>{{ result.meta.currency }}</span>
        </div>
      </div>

      <div v-if="result.issues.length" class="ds-issue-list issue-list--global">
        <span
          v-for="issue in result.issues"
          :key="`${issue.code}-${issue.path ?? issue.message}`"
          :class="issueClass(issue.level)"
        >
          {{ issue.code }}: {{ issue.message }}
        </span>
      </div>
    </section>

    <section v-if="result" class="items" data-testid="items-list">
      <article v-for="item in items" :key="item.id" class="item-card ds-card">
        <div class="item-card__top">
          <div>
            <div class="item-card__meta ds-label">Line {{ item.source.line }}</div>
            <div class="item-card__raw">{{ item.source.raw }}</div>
          </div>

          <div class="confidence-box">
            <span>Overall {{ formatConfidence(item.confidence.overall) }}</span>
            <span>Category {{ formatConfidence(item.category.confidence) }}</span>
            <span>Name {{ formatConfidence(item.confidence.fields?.name) }}</span>
            <span>Price {{ formatConfidence(item.confidence.fields?.prices) }}</span>
            <span>Size {{ formatConfidence(item.confidence.fields?.sizes) }}</span>
          </div>
        </div>

        <div class="ds-grid ds-grid--two">
          <label class="ds-field">
            <span class="ds-label">Kind</span>
            <select
              v-model="item.kind"
              class="ds-control"
              :data-testid="`kind-${item.id}`"
            >
              <option v-for="kind in KIND_OPTIONS" :key="kind" :value="kind">
                {{ kind }}
              </option>
            </select>
          </label>

          <label class="ds-field">
            <span class="ds-label">Category</span>
            <select
              class="ds-control"
              :value="item.category.label ?? ''"
              :data-testid="`category-${item.id}`"
              @change="onCategoryChange(item, $event)"
            >
              <option value="">unassigned</option>
              <option
                v-for="category in CATEGORY_LABELS"
                :key="category"
                :value="category"
              >
                {{ category }}
              </option>
            </select>
          </label>
        </div>

        <div class="ds-grid ds-grid--two">
          <label class="ds-field">
            <span class="ds-label">Name</span>
            <input
              class="ds-control"
              :value="item.fields.name ?? ''"
              :data-testid="`name-${item.id}`"
              type="text"
              @input="onNameInput(item, $event)"
            />
          </label>

          <label class="ds-field ds-field--stacked">
            <span class="ds-label">Description</span>
            <textarea
              class="ds-control"
              :value="item.fields.description ?? ''"
              :data-testid="`description-${item.id}`"
              rows="2"
              @input="onDescriptionInput(item, $event)"
            />
          </label>
        </div>

        <div class="ds-grid ds-grid--two">
          <div class="field-group">
            <div class="field-group__header">
              <span class="ds-label">First price</span>
              <button
                v-if="item.fields.prices[0]"
                class="ds-button ds-button--ghost"
                type="button"
                @click="removePrice(item)"
              >
                Remove
              </button>
              <button
                v-else
                class="ds-button ds-button--ghost"
                type="button"
                @click="addPrice(item)"
              >
                Add
              </button>
            </div>

            <div v-if="item.fields.prices[0]" class="ds-grid ds-grid--two">
              <label class="ds-field">
                <span class="ds-label">Value</span>
                <input
                  class="ds-control"
                  :value="item.fields.prices[0].value"
                  :data-testid="`price-value-${item.id}`"
                  type="number"
                  step="0.01"
                  min="0"
                  @input="onPriceValueInput(item, $event)"
                />
              </label>

              <label class="ds-field">
                <span class="ds-label">Currency</span>
                <select
                  class="ds-control"
                  :value="item.fields.prices[0].currency"
                  :data-testid="`price-currency-${item.id}`"
                  @change="onPriceCurrencyChange(item, $event)"
                >
                  <option
                    v-for="currency in CURRENCY_OPTIONS"
                    :key="currency"
                    :value="currency"
                  >
                    {{ currency }}
                  </option>
                </select>
              </label>
            </div>
          </div>

          <div class="field-group">
            <div class="field-group__header">
              <span class="ds-label">First size</span>
              <button
                v-if="item.fields.sizes[0]"
                class="ds-button ds-button--ghost"
                type="button"
                @click="removeSize(item)"
              >
                Remove
              </button>
              <button
                v-else
                class="ds-button ds-button--ghost"
                type="button"
                @click="addSize(item)"
              >
                Add
              </button>
            </div>

            <div v-if="item.fields.sizes[0]" class="ds-grid ds-grid--two">
              <label class="ds-field">
                <span class="ds-label">Value</span>
                <input
                  class="ds-control"
                  :value="item.fields.sizes[0].value"
                  :data-testid="`size-value-${item.id}`"
                  type="number"
                  step="0.01"
                  min="0"
                  @input="onSizeValueInput(item, $event)"
                />
              </label>

              <label class="ds-field">
                <span class="ds-label">Unit</span>
                <select
                  class="ds-control"
                  :value="item.fields.sizes[0].unit"
                  :data-testid="`size-unit-${item.id}`"
                  @change="onSizeUnitChange(item, $event)"
                >
                  <option v-for="unit in SIZE_UNIT_OPTIONS" :key="unit" :value="unit">
                    {{ unit }}
                  </option>
                </select>
              </label>
            </div>
          </div>
        </div>

        <div v-if="item.issues.length" class="ds-issue-list">
          <span
            v-for="issue in item.issues"
            :key="`${issue.code}-${issue.path ?? issue.message}`"
            :class="issueClass(issue.level)"
          >
            {{ issue.code }}: {{ issue.message }}
          </span>
        </div>
      </article>
    </section>
  </main>
</template>

<style scoped>
.page {
  width: min(var(--ds-page-max-width), calc(100% - var(--ds-page-gutter)));
  margin: 0 auto;
  padding: var(--ds-page-padding-top) 0 var(--ds-page-padding-bottom);
}

.panel {
  padding: var(--ds-space-9);
}

.panel + .panel,
.panel + .items,
.items + .panel {
  margin-top: var(--ds-space-7);
}

.panel--compact {
  padding: var(--ds-space-7) var(--ds-space-8);
}

.hero {
  display: grid;
  gap: var(--ds-space-8);
}

.hero__meta {
  display: flex;
  flex-wrap: wrap;
  gap: var(--ds-space-3);
}

.panel__header {
  display: flex;
  align-items: start;
  justify-content: space-between;
  gap: var(--ds-space-7);
  margin-bottom: var(--ds-space-6);
}

.error {
  margin: var(--ds-space-6) 0 0;
  color: var(--ds-color-danger);
  font-weight: 600;
}

.summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: var(--ds-space-4);
}

.summary > div {
  display: flex;
  flex-direction: column;
  gap: var(--ds-space-1);
}

.items {
  display: grid;
  gap: var(--ds-space-6);
}

.item-card {
  padding: var(--ds-space-8);
}

.item-card__top {
  display: flex;
  justify-content: space-between;
  gap: var(--ds-space-6);
  margin-bottom: var(--ds-space-6);
}

.item-card__raw {
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--ds-color-text);
  word-break: break-word;
}

.confidence-box {
  display: grid;
  gap: var(--ds-space-1);
  min-width: 150px;
  color: var(--ds-color-text-muted);
  font-size: var(--ds-font-size-small);
  text-align: right;
}

.ds-grid + .ds-grid,
.ds-grid + .ds-issue-list,
.ds-issue-list + .ds-grid {
  margin-top: var(--ds-space-4);
}

.field-group {
  border: 1px solid var(--ds-color-border);
  border-radius: var(--ds-radius-group);
  padding: var(--ds-space-5);
  background: var(--ds-color-surface-soft);
}

.field-group__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--ds-space-4);
  margin-bottom: var(--ds-space-4);
}

.issue-list--global {
  margin-top: var(--ds-space-6);
}

@media (max-width: 840px) {
  .page {
    width: min(100% - var(--ds-page-gutter-mobile), var(--ds-page-max-width));
    padding-top: var(--ds-space-7);
  }

  .panel,
  .item-card {
    padding: var(--ds-space-7);
  }

  .panel__header,
  .item-card__top {
    flex-direction: column;
  }

  .confidence-box {
    text-align: left;
  }
}
</style>
