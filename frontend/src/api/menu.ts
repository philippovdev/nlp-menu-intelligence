export type SchemaVersion = "v1";

export type IssueLevel = "info" | "warning" | "error";

export type IssueV1 = {
  code: string;
  level: IssueLevel;
  message: string;
  path?: string;
  details?: Record<string, unknown>;
};

export type CategoryPredictionV1 = {
  label: string | null;
  confidence: number | null;
};

export type PriceV1 = {
  value: number;
  currency: string;
  raw?: string;
};

export type SizeV1 = {
  value: number;
  unit: string;
  raw?: string;
};

export type ExtractedFieldsV1 = {
  name: string | null;
  description: string | null;
  prices: PriceV1[];
  sizes: SizeV1[];
};

export type ConfidenceV1 = {
  overall?: number | null;
  category?: number | null;
  fields?: Partial<Record<keyof ExtractedFieldsV1, number | null>>;
};

export type ItemKindV1 = "menu_item" | "category_header" | "noise";

export type SourceV1 = {
  line: number;
  raw: string;
};

export type MenuItemV1 = {
  id: string;
  source: SourceV1;
  kind: ItemKindV1;
  category: CategoryPredictionV1;
  fields: ExtractedFieldsV1;
  confidence: ConfidenceV1;
  issues: IssueV1[];
};

export type ModelVersionV1 = {
  category_model: string;
  ner_model: string;
};

export type ParseMenuMetaV1 = {
  lang: string;
  currency: string;
  split_strategy: "lines";
};

export type ParseMenuDocumentV1 = {
  source_type: "pdf" | "image";
  filename?: string | null;
  media_type?: string | null;
  ocr_used?: boolean | null;
  extracted_text?: string | null;
};

export type ParseMenuRequestV1 = {
  schema_version: SchemaVersion;
  text: string;
  lang?: string;
  currency_hint?: string;
  category_labels?: string[];
};

export type ParseMenuFileRequestV1 = {
  schema_version: SchemaVersion;
  file: File;
  lang?: string;
  currency_hint?: string;
  category_labels?: string[];
};

export type ParseMenuResultV1 = {
  schema_version: SchemaVersion;
  request_id: string;
  meta: ParseMenuMetaV1;
  model_version: ModelVersionV1;
  items: MenuItemV1[];
  issues: IssueV1[];
  document?: ParseMenuDocumentV1;
};

export type ParseMenuResponseV1 = ParseMenuResultV1;

export type ParseMenuFileResponseV1 = ParseMenuResultV1;

export type ApiErrorResponseV1 = {
  schema_version: SchemaVersion;
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
};

export class ApiError extends Error {
  code?: string;
  status: number;
  details?: Record<string, unknown>;

  constructor(
    message: string,
    status: number,
    code?: string,
    details?: Record<string, unknown>,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

async function readParseResult<T extends ParseMenuResultV1>(
  response: Response,
): Promise<T> {
  const data = (await response.json().catch(() => null)) as
    | ParseMenuResultV1
    | ApiErrorResponseV1
    | null;

  if (!response.ok) {
    throw new ApiError(
      data && "error" in data ? data.error.message : "Unable to parse menu.",
      response.status,
      data && "error" in data ? data.error.code : undefined,
      data && "error" in data ? data.error.details : undefined,
    );
  }

  if (!data || !("items" in data)) {
    throw new ApiError("Unexpected response from server.", response.status);
  }

  return data as T;
}

export async function parseMenu(
  payload: ParseMenuRequestV1,
): Promise<ParseMenuResponseV1> {
  const response = await fetch("/api/v1/menu/parse", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  return readParseResult<ParseMenuResponseV1>(response);
}

export async function parseMenuFile(
  payload: ParseMenuFileRequestV1,
): Promise<ParseMenuFileResponseV1> {
  const formData = new FormData();

  formData.append("schema_version", payload.schema_version);
  formData.append("file", payload.file);

  if (payload.lang) {
    formData.append("lang", payload.lang);
  }

  if (payload.currency_hint) {
    formData.append("currency_hint", payload.currency_hint);
  }

  payload.category_labels?.forEach((label) => {
    formData.append("category_labels", label);
  });

  const response = await fetch("/api/v1/menu/parse-file", {
    method: "POST",
    body: formData,
  });

  return readParseResult<ParseMenuFileResponseV1>(response);
}
