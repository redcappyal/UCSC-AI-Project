import SwiftUI
import WebKit

/// The existing web product inside the native shell. Remote by design:
/// web fixes ship without an app update (spec decision).
struct WebScreen: View {
    let url: URL
    var showsClose = false
    @Environment(\.dismiss) private var dismiss
    // The page's own theme (localStorage 'slc-theme', independent of the
    // device appearance), reported by the injected observer below. Driving
    // preferredColorScheme with it is what keeps the status bar text legible
    // now that the page paints full-bleed behind it.
    @State private var webTheme: ColorScheme?

    var body: some View {
        ZStack(alignment: .topTrailing) {
            // Full-bleed on every edge: the page carries viewport-fit=cover
            // and pads its own header/dock with env(safe-area-inset-*), so
            // insetting it here would double the clearance and paint a native
            // band (the "black notch") above the web background.
            WebViewRepresentable(url: url) { webTheme = $0 }
                .ignoresSafeArea()
            if showsClose {
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title)
                        .foregroundStyle(Theme.dim)
                        .padding(12)
                }
                .accessibilityLabel("Close review")
            }
        }
        // System background, not Theme.bg: this shows behind the page before
        // first paint and must track the device appearance the way the web
        // theme default does — a pinned black backing reads as a broken
        // status bar over light-mode web content.
        .background(Color(.systemBackground))
        .preferredColorScheme(webTheme)
    }
}

private struct WebViewRepresentable: UIViewRepresentable {
    let url: URL
    let onThemeChange: (ColorScheme?) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(onThemeChange: onThemeChange) }

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.allowsInlineMediaPlayback = true
        // Mirrors index.html's boot script: <html data-theme> is set before
        // first paint and rewritten by applyTheme(), so observing that one
        // attribute covers both the initial theme and later toggles.
        let observer = WKUserScript(
            source: """
            (function () {
              const report = () => window.webkit.messageHandlers.slcTheme
                .postMessage(document.documentElement.dataset.theme || '');
              new MutationObserver(report).observe(
                document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
              report();
            })();
            """,
            injectionTime: .atDocumentStart,
            forMainFrameOnly: true)
        configuration.userContentController.addUserScript(observer)
        configuration.userContentController.add(context.coordinator, name: "slcTheme")
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.isOpaque = false
        webView.backgroundColor = .clear
        // The page is a fixed-viewport shell (body never scrolls; sections
        // scroll internally), so the outer scroll view has no legitimate
        // movement. Left with the automatic safe-area insets it becomes
        // draggable anyway, letting the whole page rubber-band sideways.
        webView.scrollView.contentInsetAdjustmentBehavior = .never
        webView.scrollView.bounces = false
        #if DEBUG
        webView.isInspectable = true
        #endif
        webView.load(URLRequest(url: url))
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        // Loaded once in makeUIView; SwiftUI re-renders must not reload the page.
    }

    static func dismantleUIView(_ webView: WKWebView, coordinator: Coordinator) {
        // The handler retains the coordinator; without this a torn-down tab
        // (the .id(serverBase) reload) leaks its coordinator and webview.
        webView.configuration.userContentController.removeScriptMessageHandler(forName: "slcTheme")
    }

    final class Coordinator: NSObject, WKScriptMessageHandler {
        let onThemeChange: (ColorScheme?) -> Void
        init(onThemeChange: @escaping (ColorScheme?) -> Void) {
            self.onThemeChange = onThemeChange
        }

        func userContentController(_ userContentController: WKUserContentController,
                                   didReceive message: WKScriptMessage) {
            switch message.body as? String {
            case "light": onThemeChange(.light)
            case "dark": onThemeChange(.dark)
            default: onThemeChange(nil)
            }
        }
    }
}
