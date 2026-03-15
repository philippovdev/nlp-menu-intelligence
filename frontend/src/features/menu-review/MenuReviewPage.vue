<script setup lang="ts">
import { computed, ref } from "vue";
import {
  ApiError,
  parseMenu,
  parseMenuFile,
  type IssueLevel,
  type MenuItemV1,
  type ParseMenuResultV1,
} from "../../api/menu";
import {
  CATEGORY_LABELS,
  CURRENCY_OPTIONS,
  DEFAULT_PARSE_REQUEST,
  KIND_OPTIONS,
  SIZE_UNIT_OPTIONS,
} from "./schema";

const supportedFileTypes = [
  "application/pdf",
  "image/jpeg",
  "image/png",
  "image/webp",
];
const acceptedFileTypes = supportedFileTypes.join(",");
const supportedFileTypeSet = new Set<string>(supportedFileTypes);
const supportedFileExtensions = [".pdf", ".jpg", ".jpeg", ".png", ".webp"];
const lowConfidenceThreshold = 0.8;

const text = ref("");
const selectedFile = ref<File | null>(null);
const fileInput = ref<HTMLInputElement | null>(null);
const loading = ref(false);
const errorMessage = ref("");
const result = ref<ParseMenuResultV1 | null>(null);

const hasSelectedFile = computed(() => selectedFile.value !== null);
const canSubmit = computed(
  () =>
    !loading.value &&
    (hasSelectedFile.value || text.value.trim().length > 0),
);
const items = computed(() => result.value?.items ?? []);
const parseButtonLabel = computed(() => {
  if (loading.value) {
    return hasSelectedFile.value ? "Parsing file..." : "Parsing menu...";
  }

  return hasSelectedFile.value ? "Parse file" : "Parse menu";
});
const selectedFileName = computed(
  () => selectedFile.value?.name ?? "No file selected",
);
const documentMetadata = computed(() => result.value?.document ?? null);
const sourceLabel = computed(() =>
  documentMetadata.value?.source_type
    ? formatSourceType(documentMetadata.value.source_type)
    : "Text",
);
const flaggedItemsCount = computed(() =>
  items.value.filter((item) => needsAttention(item)).length,
);
const extractedTextPreview = computed(() => {
  const value = result.value?.document?.extracted_text?.trim();
  return value ? value : null;
});
const groupedItems = computed(() => {
  const groups = new Map<
    string,
    {
      key: string;
      title: string;
      items: MenuItemV1[];
      flaggedCount: number;
    }
  >();

  items.value.forEach((item) => {
    const key =
      item.category.label ??
      (item.kind === "noise" ? "noise" : "unassigned");
    const group =
      groups.get(key) ??
      {
        key,
        title: formatGroupTitle(key),
        items: [],
        flaggedCount: 0,
      };

    group.items.push(item);

    if (needsAttention(item)) {
      group.flaggedCount += 1;
    }

    groups.set(key, group);
  });

  return [...groups.values()].sort((left, right) => {
    return getGroupOrder(left.key) - getGroupOrder(right.key);
  });
});
const groupCount = computed(() => groupedItems.value.length);

function isSupportedFile(file: File): boolean {
  if (supportedFileTypeSet.has(file.type)) {
    return true;
  }

  const name = file.name.toLowerCase();
  return supportedFileExtensions.some((extension) => name.endsWith(extension));
}

function clearSelectedFile(): void {
  selectedFile.value = null;

  if (fileInput.value) {
    fileInput.value.value = "";
  }
}

function onFileChange(event: Event): void {
  const target = event.target as HTMLInputElement;
  const file = target.files?.[0] ?? null;

  if (!file) {
    selectedFile.value = null;
    return;
  }

  if (!isSupportedFile(file)) {
    clearSelectedFile();
    errorMessage.value = "Unsupported file type. Use PDF, JPG, PNG, or WEBP.";
    return;
  }

  selectedFile.value = file;
  errorMessage.value = "";
}

function formatConfidence(value?: number | null): string {
  if (typeof value !== "number") {
    return "-";
  }

  return `${Math.round(value * 100)}%`;
}

function getConfidenceValues(item: MenuItemV1): number[] {
  return [
    item.confidence.overall,
    item.category.confidence,
    item.confidence.fields?.name,
    item.confidence.fields?.prices,
    item.confidence.fields?.sizes,
  ].filter((value): value is number => typeof value === "number");
}

function isLowConfidence(item: MenuItemV1): boolean {
  return getConfidenceValues(item).some((value) => value < lowConfidenceThreshold);
}

function needsAttention(item: MenuItemV1): boolean {
  return item.issues.some((issue) => issue.level !== "info") || isLowConfidence(item);
}

function hasIssueLevel(item: MenuItemV1, level: IssueLevel): boolean {
  return item.issues.some((issue) => issue.level === level);
}

function issueClass(level: IssueLevel): string {
  return `ds-issue ds-issue--${level}`;
}

function itemCardClass(item: MenuItemV1): string[] {
  return [
    "item-card",
    "ds-card",
    hasIssueLevel(item, "error")
      ? "item-card--error"
      : needsAttention(item)
        ? "item-card--warning"
        : undefined,
  ].filter((value): value is string => typeof value === "string");
}

function getGroupOrder(key: string): number {
  const labelIndex = CATEGORY_LABELS.indexOf(key as (typeof CATEGORY_LABELS)[number]);

  if (labelIndex >= 0) {
    return labelIndex;
  }

  if (key === "unassigned") {
    return CATEGORY_LABELS.length + 1;
  }

  if (key === "noise") {
    return CATEGORY_LABELS.length + 2;
  }

  return CATEGORY_LABELS.length + 3;
}

function formatGroupTitle(key: string): string {
  if (key === "unassigned") {
    return "Unassigned";
  }

  if (key === "noise") {
    return "Noise";
  }

  return key
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatSourceType(value: string): string {
  if (value === "pdf") {
    return "PDF";
  }

  if (value === "image") {
    return "Image";
  }

  return value.charAt(0).toUpperCase() + value.slice(1);
}

function toNullable(value: string): string | null {
  const next = value.trim();
  return next.length > 0 ? next : null;
}

function buildExportFilename(extension: "csv" | "json"): string {
  const sourceName =
    result.value?.document?.filename?.replace(/\.[^.]+$/, "") ?? "menu-review";
  const safeName =
    sourceName
      .trim()
      .replace(/[^a-zA-Z0-9_-]+/g, "-")
      .replace(/^-+|-+$/g, "") || "menu-review";

  return `${safeName}-${result.value?.request_id ?? "export"}.${extension}`;
}

function downloadFile(
  content: string,
  type: string,
  extension: "csv" | "json",
): void {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.href = url;
  link.download = buildExportFilename(extension);
  document.body.appendChild(link);
  link.click();
  link.remove();

  setTimeout(() => {
    URL.revokeObjectURL(url);
  }, 0);
}

function exportJson(): void {
  if (!result.value) {
    return;
  }

  downloadFile(
    JSON.stringify(result.value, null, 2),
    "application/json",
    "json",
  );
}

function escapeCsvValue(value: number | string | null | undefined): string {
  const normalized = value == null ? "" : String(value);
  return `"${normalized.replace(/"/g, '""')}"`;
}

function exportCsv(): void {
  if (!result.value) {
    return;
  }

  const header = [
    "id",
    "line",
    "raw",
    "kind",
    "category_label",
    "category_confidence",
    "name",
    "description",
    "price_value",
    "price_currency",
    "size_value",
    "size_unit",
    "overall_confidence",
    "issue_codes",
  ];
  const rows = result.value.items.map((item) => {
    const firstPrice = item.fields.prices[0];
    const firstSize = item.fields.sizes[0];

    return [
      item.id,
      item.source.line,
      item.source.raw,
      item.kind,
      item.category.label,
      item.category.confidence,
      item.fields.name,
      item.fields.description,
      firstPrice?.value,
      firstPrice?.currency,
      firstSize?.value,
      firstSize?.unit,
      item.confidence.overall,
      item.issues.map((issue) => issue.code).join("|"),
    ]
      .map((value) => escapeCsvValue(value))
      .join(",");
  });

  downloadFile(
    [`\uFEFF${header.join(",")}`, ...rows].join("\n"),
    "text/csv;charset=utf-8",
    "csv",
  );
}

async function submit(): Promise<void> {
  if (!canSubmit.value) {
    return;
  }

  loading.value = true;
  errorMessage.value = "";

  try {
    const requestOptions = {
      schema_version: DEFAULT_PARSE_REQUEST.schema_version,
      lang: DEFAULT_PARSE_REQUEST.lang,
      currency_hint: DEFAULT_PARSE_REQUEST.currency_hint,
      category_labels: [...CATEGORY_LABELS],
    };
    const response = selectedFile.value
      ? await parseMenuFile({
          ...requestOptions,
          file: selectedFile.value,
        })
      : await parseMenu({
          ...requestOptions,
          text: text.value,
        });

    result.value = structuredClone(response);
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
        <p class="ds-label ds-eyebrow">Slice 2 Intake + Review</p>
        <h1 class="ds-title ds-title--hero">
          Intake can start from pasted text or a single uploaded file.
        </h1>
        <p class="ds-lead">
          Keep one review screen, route both intake paths through the backend,
          and land in the same structured editing state.
        </p>
      </div>
      <div class="hero__meta">
        <span class="ds-pill">POST /api/v1/menu/parse</span>
        <span class="ds-pill">POST /api/v1/menu/parse-file</span>
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
          {{ parseButtonLabel }}
        </button>
      </div>

      <div class="intake-grid">
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

        <div class="file-intake">
          <label class="ds-field">
            <span class="ds-label">Upload one file</span>
            <input
              ref="fileInput"
              class="ds-control"
              data-testid="menu-file"
              type="file"
              :accept="acceptedFileTypes"
              @change="onFileChange"
            />
          </label>

          <div class="file-state">
            <div>
              <span class="ds-label">Selected file</span>
              <div class="file-name" data-testid="selected-file-name">
                {{ selectedFileName }}
              </div>
            </div>
            <button
              v-if="hasSelectedFile"
              class="ds-button ds-button--ghost"
              type="button"
              @click="clearSelectedFile"
            >
              Clear
            </button>
          </div>

          <p class="input-note">
            Accepted formats: PDF, JPG, PNG, WEBP. If a file is selected, it is
            parsed instead of the pasted text.
          </p>
        </div>
      </div>

      <p v-if="errorMessage" class="error">{{ errorMessage }}</p>
    </section>

    <section v-if="result" class="panel panel--compact ds-card">
      <div class="result-toolbar">
        <div class="summary">
          <div>
            <span class="ds-label">Items</span>
            <span>{{ result.items.length }}</span>
          </div>
          <div>
            <span class="ds-label">Groups</span>
            <span>{{ groupCount }}</span>
          </div>
          <div>
            <span class="ds-label">Need review</span>
            <span>{{ flaggedItemsCount }}</span>
          </div>
          <div>
            <span class="ds-label">Source</span>
            <span>{{ sourceLabel }}</span>
          </div>
          <div>
            <span class="ds-label">Lang</span>
            <span>{{ result.meta.lang }}</span>
          </div>
          <div>
            <span class="ds-label">Currency</span>
            <span>{{ result.meta.currency }}</span>
          </div>
          <div>
            <span class="ds-label">Category model</span>
            <span class="summary__value summary__value--meta">
              {{ result.model_version.category_model }}
            </span>
          </div>
        </div>

        <div class="result-actions">
          <button
            class="ds-button ds-button--ghost"
            data-testid="export-json-button"
            type="button"
            @click="exportJson"
          >
            Export JSON
          </button>
          <button
            class="ds-button ds-button--ghost"
            data-testid="export-csv-button"
            type="button"
            @click="exportCsv"
          >
            Export CSV
          </button>
        </div>
      </div>

      <div class="summary summary--document">
        <div>
          <span class="ds-label">Request</span>
          <span class="summary__value summary__value--meta">
            {{ result.request_id }}
          </span>
        </div>
        <div>
          <span class="ds-label">Fields parser</span>
          <span class="summary__value summary__value--meta">
            {{ result.model_version.ner_model }}
          </span>
        </div>
        <div v-if="documentMetadata?.filename">
          <span class="ds-label">Filename</span>
          <span>{{ documentMetadata.filename }}</span>
        </div>
        <div v-if="documentMetadata?.ocr_used != null">
          <span class="ds-label">OCR</span>
          <span>{{ documentMetadata.ocr_used ? "used" : "not used" }}</span>
        </div>
      </div>

      <details v-if="extractedTextPreview" class="document-preview">
        <summary class="ds-label">Extracted text</summary>
        <pre>{{ extractedTextPreview }}</pre>
      </details>

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

    <section
      v-if="result"
      class="review-groups"
      data-testid="items-list"
    >
      <section
        v-for="group in groupedItems"
        :key="group.key"
        class="group-section"
        :data-testid="`group-${group.key}`"
      >
        <div class="group-header ds-card">
          <div>
            <p class="ds-label ds-eyebrow">Grouped review</p>
            <h3 class="group-title">{{ group.title }}</h3>
          </div>

          <div class="group-meta">
            <span class="ds-pill">{{ group.items.length }} items</span>
            <span
              v-if="group.flaggedCount"
              class="ds-issue ds-issue--warning"
            >
              {{ group.flaggedCount }} need attention
            </span>
          </div>
        </div>

        <div class="group-items">
          <article
            v-for="item in group.items"
            :key="item.id"
            :class="itemCardClass(item)"
          >
            <div class="item-card__top">
              <div>
                <div class="item-card__meta ds-label">Line {{ item.source.line }}</div>
                <div class="item-card__raw">{{ item.source.raw }}</div>
              </div>

              <div class="item-status">
                <div class="item-attention">
                  <span
                    v-if="hasIssueLevel(item, 'error')"
                    class="ds-issue ds-issue--error"
                  >
                    Error
                  </span>
                  <span
                    v-else-if="item.issues.some((issue) => issue.level === 'warning')"
                    class="ds-issue ds-issue--warning"
                  >
                    Warning
                  </span>
                  <span
                    v-if="isLowConfidence(item)"
                    class="ds-issue ds-issue--warning"
                    :data-testid="`low-confidence-${item.id}`"
                  >
                    Low confidence
                  </span>
                </div>

                <div class="confidence-box">
                  <span>Overall {{ formatConfidence(item.confidence.overall) }}</span>
                  <span>Category {{ formatConfidence(item.category.confidence) }}</span>
                  <span>Name {{ formatConfidence(item.confidence.fields?.name) }}</span>
                  <span>Price {{ formatConfidence(item.confidence.fields?.prices) }}</span>
                  <span>Size {{ formatConfidence(item.confidence.fields?.sizes) }}</span>
                </div>
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
                  <option
                    value=""
                  >
                    unassigned
                  </option>
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
        </div>
      </section>
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

.intake-grid {
  display: grid;
  gap: var(--ds-space-6);
}

.file-intake {
  display: grid;
  gap: var(--ds-space-4);
  padding: var(--ds-space-5);
  border: 1px solid var(--ds-color-border);
  border-radius: var(--ds-radius-group);
  background: var(--ds-color-surface-soft);
}

.file-state {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: var(--ds-space-4);
}

.file-name {
  margin-top: var(--ds-space-2);
  color: var(--ds-color-text);
  font-weight: 600;
  word-break: break-word;
}

.input-note {
  margin: 0;
  color: var(--ds-color-text-muted);
  font-size: var(--ds-font-size-small);
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

.summary__value {
  word-break: break-word;
}

.summary__value--meta {
  color: var(--ds-color-text-soft);
  font-size: var(--ds-font-size-small);
}

.result-toolbar {
  display: grid;
  gap: var(--ds-space-6);
}

.result-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: var(--ds-space-3);
}

.summary--document {
  margin-top: var(--ds-space-6);
}

.document-preview {
  margin-top: var(--ds-space-6);
}

.document-preview summary {
  cursor: pointer;
}

.document-preview pre {
  margin: var(--ds-space-4) 0 0;
  padding: var(--ds-space-5);
  border: 1px solid var(--ds-color-border);
  border-radius: var(--ds-radius-group);
  background: var(--ds-color-surface-soft);
  color: var(--ds-color-text);
  font-family: inherit;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
}

.review-groups {
  display: grid;
  gap: var(--ds-space-6);
}

.group-section {
  display: grid;
  gap: var(--ds-space-4);
}

.group-header {
  display: flex;
  align-items: start;
  justify-content: space-between;
  gap: var(--ds-space-6);
  padding: var(--ds-space-7);
}

.group-title {
  margin: 0;
  color: var(--ds-color-text);
  font-family: var(--ds-font-display);
  font-size: 1.4rem;
}

.group-meta {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: var(--ds-space-2);
}

.group-items {
  display: grid;
  gap: var(--ds-space-4);
}

.item-card {
  padding: var(--ds-space-8);
}

.item-card--warning {
  border-color: var(--ds-color-warning-bg);
}

.item-card--error {
  border-color: var(--ds-color-error-bg);
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

.item-status {
  display: grid;
  gap: var(--ds-space-3);
}

.item-attention {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: var(--ds-space-2);
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
  .group-header,
  .item-card__top {
    flex-direction: column;
  }

  .file-state {
    align-items: start;
    flex-direction: column;
  }

  .group-meta,
  .item-attention,
  .result-actions {
    justify-content: flex-start;
  }

  .confidence-box {
    text-align: left;
  }
}
</style>
