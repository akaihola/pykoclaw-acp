import { test, expect } from "@playwright/test";

test("ACP connection to Mitto", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("body")).toBeVisible();
});
