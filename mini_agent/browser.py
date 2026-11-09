import base64
from playwright.async_api import(
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)
from mini_agent.schemas import (
    BrowserState,
    PageElement,
)

class BrowserSession:
    def __init__(self,headless:bool=False):
        self.headless=headless
        self._playwright:Playwright|None=None
        self._browser:Browser|None=None
        self._context:BrowserContext|None=None
        self._page:Page|None=None

    @property
    def page(self)->Page:
        if self._page is None:
            raise RuntimeError("浏览器尚未启动")
        return self._page

    async def start(self)->None:
        self._playwright=await async_playwright().start()
        self._browser=await self._playwright.chromium.launch(
            headless=self.headless
        )
        self._context=await self._browser.new_context(
            viewport={
                "width":1280,
                "height":900,
            }
        )
        self._page=await self._context.new_page()

    async def close(self)->None:
        if self._context is not None:
            await self._context.close()
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()
        self._page=None
        self._context=None
        self._browser=None
        self._playwright=None

    async def goto(self,url:str)->None:
        await self.page.goto(
            url,
            wait_until="domcontentloaded"
        )

    async def get_state(self)->BrowserState:
        raw_elements=await self.page.evaluate(
            """
            ()=>{
            const selector=[
                "a",
                "button",
                "input",
                "textarea",
                "select",
                "[role='button']",
                "[contenteditable='true']"
            ].join(",");
            
            const allElements=Array.from(
                document.querySelectorAll(selector)
            );
                const visibleElements = allElements.filter(
                    (element) => {
                        const style = window.getComputedStyle(element);
                        const rect = element.getBoundingClientRect();

                        return (
                            style.display !== "none" &&
                            style.visibility !== "hidden" &&
                            rect.width > 0 &&
                            rect.height > 0
                        );
                    }
                );

                return visibleElements
                    .slice(0, 80)
                    .map((element, index) => {
                        element.setAttribute(
                            "data-agent-index",
                            String(index)
                        );

                        const rawText = (
                            element.innerText ||
                            element.value ||
                            element.getAttribute("aria-label") ||
                            element.getAttribute("placeholder") ||
                            ""
                        );

                        const text = rawText
                            .trim()
                            .replace(/\\s+/g, " ")
                            .slice(0, 200);

                        return {
                            index: index,
                            tag: element.tagName.toLowerCase(),
                            text: text,
                            element_type:
                                element.getAttribute("type"),
                            placeholder:
                                element.getAttribute("placeholder"),
                            aria_label:
                                element.getAttribute("aria-label")
                        };
                    });
            }
            """
        )

        visible_text=await self.page.locator(
            "body"
        ).inner_text()
        visible_text=visible_text[:5000]

        screenshot_bytes=await self.page.screenshot(
            full_page=False
        )

        screenshot_base64=base64.b64encode(
            screenshot_bytes
        ).decode("utf-8")

        elements=[
            PageElement.model_validate(element)
            for element in raw_elements
        ]

        return BrowserState(
            url=self.page.url,
            title=await self.page.title(),
            visible_text=visible_text,
            elements=elements,
            screenshot_base64=screenshot_base64
        )
            
            
































































































