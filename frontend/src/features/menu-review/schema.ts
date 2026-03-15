export const CATEGORY_LABELS = [
  "salads",
  "soups",
  "mains",
  "pizza",
  "pasta",
  "burgers",
  "sides",
  "desserts",
  "breakfast",
  "drinks_hot",
  "drinks_cold",
  "other",
] as const;

export const KIND_OPTIONS = [
  "menu_item",
  "category_header",
  "noise",
] as const;

export const CURRENCY_OPTIONS = ["RUB", "USD", "EUR"] as const;

export const SIZE_UNIT_OPTIONS = ["g", "kg", "ml", "l", "pcs"] as const;

export const DEFAULT_PARSE_REQUEST = {
  schema_version: "v1" as const,
  lang: "ru",
  currency_hint: "RUB",
};
