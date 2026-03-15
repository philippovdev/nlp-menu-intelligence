import { mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import type {
  ApiErrorResponseV1,
  ParseMenuFileResponseV1,
  ParseMenuResponseV1,
} from "../../api/menu";
import { CATEGORY_LABELS } from "./schema";
import MenuReviewPage from "./MenuReviewPage.vue";

const mockResponse: ParseMenuResponseV1 = {
  schema_version: "v1",
  request_id: "req_test_001",
  meta: {
    lang: "ru",
    currency: "RUB",
    split_strategy: "lines",
  },
  model_version: {
    category_model: "stub@0.0.0",
    ner_model: "stub@0.0.0",
  },
  items: [
    {
      id: "item_1",
      source: {
        line: 1,
        raw: "САЛАТЫ",
      },
      kind: "category_header",
      category: {
        label: "salads",
        confidence: 0.87,
      },
      fields: {
        name: null,
        description: null,
        prices: [],
        sizes: [],
      },
      confidence: {
        overall: 0.87,
        category: 0.87,
        fields: {},
      },
      issues: [],
    },
    {
      id: "item_2",
      source: {
        line: 2,
        raw: "Цезарь с курицей 250 г - 390 ₽",
      },
      kind: "menu_item",
      category: {
        label: "salads",
        confidence: 0.94,
      },
      fields: {
        name: "Цезарь с курицей",
        description: null,
        prices: [{ value: 390, currency: "RUB", raw: "390 ₽" }],
        sizes: [{ value: 250, unit: "g", raw: "250 г" }],
      },
      confidence: {
        overall: 0.72,
        category: 0.74,
        fields: {
          name: 0.76,
          prices: 0.89,
          sizes: 0.93,
        },
      },
      issues: [
        {
          code: "LOW_CONFIDENCE_CATEGORY",
          level: "warning",
          message: "Category confidence below threshold.",
          path: "/items/1/category",
        },
      ],
    },
    {
      id: "item_3",
      source: {
        line: 3,
        raw: "Лимонад 330 мл - 250 ₽",
      },
      kind: "menu_item",
      category: {
        label: "drinks_cold",
        confidence: 0.97,
      },
      fields: {
        name: "Лимонад",
        description: null,
        prices: [{ value: 250, currency: "RUB", raw: "250 ₽" }],
        sizes: [{ value: 330, unit: "ml", raw: "330 мл" }],
      },
      confidence: {
        overall: 0.95,
        category: 0.97,
        fields: {
          name: 0.96,
          prices: 0.94,
          sizes: 0.95,
        },
      },
      issues: [],
    },
  ],
  issues: [
    {
      code: "EMPTY_LINES_SKIPPED",
      level: "info",
      message: "Some empty lines were ignored.",
    },
  ],
};

const mockFileResponse: ParseMenuFileResponseV1 = {
  ...mockResponse,
  request_id: "req_file_001",
  document: {
    source_type: "pdf",
    filename: "menu.pdf",
    media_type: "application/pdf",
    ocr_used: true,
    extracted_text: "САЛАТЫ\nЦезарь с курицей 250 г - 390 ₽",
  },
};

function flushPromises(): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, 0);
  });
}

function readBlobText(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(blob);
  });
}

function mockFetchSuccess(
  payload: ParseMenuResponseV1 | ParseMenuFileResponseV1,
) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    json: vi.fn().mockResolvedValue(payload),
  });

  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function mockFetchError(payload: ApiErrorResponseV1) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: false,
    status: 400,
    json: vi.fn().mockResolvedValue(payload),
  });

  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function mockDownloadApis() {
  const createObjectURL = vi.fn(() => "blob:menu-review");
  const revokeObjectURL = vi.fn();
  const clickSpy = vi
    .spyOn(HTMLAnchorElement.prototype, "click")
    .mockImplementation(() => {});

  Object.defineProperty(URL, "createObjectURL", {
    value: createObjectURL,
    configurable: true,
  });
  Object.defineProperty(URL, "revokeObjectURL", {
    value: revokeObjectURL,
    configurable: true,
  });

  return {
    clickSpy,
    createObjectURL,
    revokeObjectURL,
  };
}

async function selectFile(
  wrapper: ReturnType<typeof mount>,
  file: File,
): Promise<void> {
  const input = wrapper.get('[data-testid="menu-file"]');

  Object.defineProperty(input.element, "files", {
    value: [file],
    configurable: true,
  });

  await input.trigger("change");
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("MenuReviewPage", () => {
  it("submits pasted text and renders parsed items", async () => {
    const fetchMock = mockFetchSuccess(mockResponse);

    const wrapper = mount(MenuReviewPage);

    await wrapper
      .get('[data-testid="menu-text"]')
      .setValue("САЛАТЫ\nЦезарь с курицей 250 г - 390 ₽");
    await wrapper.get('[data-testid="parse-button"]').trigger("click");
    await flushPromises();

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/menu/parse",
      expect.objectContaining({
        method: "POST",
      }),
    );
    expect(wrapper.get('[data-testid="items-list"]').text()).toContain(
      "Цезарь с курицей 250 г - 390 ₽",
    );
    expect(wrapper.text()).toContain("EMPTY_LINES_SKIPPED");
  });

  it("updates local state when a file is selected", async () => {
    const wrapper = mount(MenuReviewPage);
    const file = new File(["pdf"], "menu.pdf", {
      type: "application/pdf",
    });

    await selectFile(wrapper, file);

    expect(wrapper.get('[data-testid="selected-file-name"]').text()).toContain(
      "menu.pdf",
    );
    expect(wrapper.get('[data-testid="parse-button"]').text()).toBe(
      "Parse file",
    );
  });

  it("sends FormData to parse-file when a file is selected", async () => {
    const fetchMock = mockFetchSuccess(mockFileResponse);
    const wrapper = mount(MenuReviewPage);
    const file = new File(["pdf"], "menu.pdf", {
      type: "application/pdf",
    });

    await selectFile(wrapper, file);
    await wrapper.get('[data-testid="parse-button"]').trigger("click");
    await flushPromises();

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/menu/parse-file",
      expect.objectContaining({
        method: "POST",
        body: expect.any(FormData),
      }),
    );

    const request = fetchMock.mock.calls[0]?.[1];
    const body = request?.body as FormData;

    expect(body.get("file")).toBe(file);
    expect(body.get("lang")).toBe("ru");
    expect(body.get("currency_hint")).toBe("RUB");
    expect(body.getAll("category_labels")).toEqual([...CATEGORY_LABELS]);
  });

  it("renders the shared review UI after a successful file parse", async () => {
    mockFetchSuccess(mockFileResponse);

    const wrapper = mount(MenuReviewPage);
    const file = new File(["pdf"], "menu.pdf", {
      type: "application/pdf",
    });

    await selectFile(wrapper, file);
    await wrapper.get('[data-testid="parse-button"]').trigger("click");
    await flushPromises();

    expect(wrapper.get('[data-testid="items-list"]').text()).toContain(
      "Цезарь с курицей 250 г - 390 ₽",
    );
    expect(wrapper.text()).toContain("Filename");
    expect(wrapper.text()).toContain("menu.pdf");
    expect(wrapper.text()).toContain("OCR");
    expect(wrapper.text()).toContain("used");
  });

  it("renders grouped review sections from the flat items response", async () => {
    mockFetchSuccess(mockResponse);

    const wrapper = mount(MenuReviewPage);

    await wrapper
      .get('[data-testid="menu-text"]')
      .setValue(
        "САЛАТЫ\nЦезарь с курицей 250 г - 390 ₽\nЛимонад 330 мл - 250 ₽",
      );
    await wrapper.get('[data-testid="parse-button"]').trigger("click");
    await flushPromises();

    expect(wrapper.get('[data-testid="group-salads"]').text()).toContain(
      "Salads",
    );
    expect(wrapper.get('[data-testid="group-drinks_cold"]').text()).toContain(
      "Drinks Cold",
    );
  });

  it("surfaces issue and low-confidence states clearly", async () => {
    mockFetchSuccess(mockResponse);

    const wrapper = mount(MenuReviewPage);

    await wrapper
      .get('[data-testid="menu-text"]')
      .setValue(
        "САЛАТЫ\nЦезарь с курицей 250 г - 390 ₽\nЛимонад 330 мл - 250 ₽",
      );
    await wrapper.get('[data-testid="parse-button"]').trigger("click");
    await flushPromises();

    expect(wrapper.get('[data-testid="low-confidence-item_2"]').text()).toBe(
      "Low confidence",
    );
    expect(wrapper.text()).toContain("LOW_CONFIDENCE_CATEGORY");
  });

  it("shows a readable error when file parsing fails", async () => {
    mockFetchError({
      schema_version: "v1",
      error: {
        code: "UNSUPPORTED_FILE",
        message: "Unable to parse file.",
      },
    });

    const wrapper = mount(MenuReviewPage);
    const file = new File(["pdf"], "menu.pdf", {
      type: "application/pdf",
    });

    await selectFile(wrapper, file);
    await wrapper.get('[data-testid="parse-button"]').trigger("click");
    await flushPromises();

    expect(wrapper.text()).toContain("Unable to parse file.");
  });

  it("exports the current edited review state as JSON", async () => {
    mockFetchSuccess(mockResponse);
    const { clickSpy, createObjectURL, revokeObjectURL } = mockDownloadApis();

    const wrapper = mount(MenuReviewPage);

    await wrapper
      .get('[data-testid="menu-text"]')
      .setValue(
        "САЛАТЫ\nЦезарь с курицей 250 г - 390 ₽\nЛимонад 330 мл - 250 ₽",
      );
    await wrapper.get('[data-testid="parse-button"]').trigger("click");
    await flushPromises();

    await wrapper.get('[data-testid="category-item_2"]').setValue("drinks_hot");
    await wrapper.get('[data-testid="name-item_2"]').setValue("Цезарь");
    await wrapper.get('[data-testid="export-json-button"]').trigger("click");
    await flushPromises();

    const exportedBlob = createObjectURL.mock.calls[0]?.[0] as Blob;
    const exportedText = await readBlobText(exportedBlob);
    const exportedPayload = JSON.parse(exportedText) as ParseMenuResponseV1;
    const editedItem = exportedPayload.items.find((item) => item.id === "item_2");

    expect(editedItem?.fields.name).toBe("Цезарь");
    expect(editedItem?.category.label).toBe("drinks_hot");
    expect(clickSpy).toHaveBeenCalled();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:menu-review");
  });

  it("exports the current reviewed items as CSV", async () => {
    mockFetchSuccess(mockResponse);
    const { createObjectURL, revokeObjectURL, clickSpy } = mockDownloadApis();

    const wrapper = mount(MenuReviewPage);

    await wrapper
      .get('[data-testid="menu-text"]')
      .setValue(
        "САЛАТЫ\nЦезарь с курицей 250 г - 390 ₽\nЛимонад 330 мл - 250 ₽",
      );
    await wrapper.get('[data-testid="parse-button"]').trigger("click");
    await flushPromises();

    await wrapper.get('[data-testid="category-item_2"]').setValue("mains");
    await wrapper.get('[data-testid="name-item_2"]').setValue("Цезарь XL");
    await wrapper.get('[data-testid="export-csv-button"]').trigger("click");
    await flushPromises();

    const exportedBlob = createObjectURL.mock.calls[0]?.[0] as Blob;
    const exportedText = await readBlobText(exportedBlob);

    expect(exportedText).toContain("category_label");
    expect(exportedText).toContain("\"mains\"");
    expect(exportedText).toContain("\"Цезарь XL\"");
    expect(exportedText).toContain("\"LOW_CONFIDENCE_CATEGORY\"");
    expect(clickSpy).toHaveBeenCalled();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:menu-review");
  });
});
