package meteordevelopment.meteorclient.utils.network;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.screen.TitleScreen;
import net.minecraft.client.gui.screen.multiplayer.ConnectScreen;
import net.minecraft.client.gui.screen.DisconnectedScreen;
import net.minecraft.client.network.ServerAddress;
import net.minecraft.client.network.ServerInfo;
import net.minecraft.client.util.ScreenshotRecorder;

public class WebCommandServer {
    private static HttpServer server;

    public static void start() {
        if (server != null) return;

        try {
            server = HttpServer.create(new InetSocketAddress("127.0.0.1", 7777), 0);

            server.createContext("/", WebCommandServer::handleRoot);
            server.createContext("/connect", WebCommandServer::handleConnect);

            // Use a dedicated non-daemon thread so the server keeps running even if
            // Minecraft changes screens or idles in the background. This ensures the
            // web command server remains available until the game exits explicitly.
            ThreadFactory factory = r -> {
                Thread t = new Thread(r, "Meteor-WebCommandServer");
                t.setDaemon(false);
                return t;
            };

            server.setExecutor(Executors.newSingleThreadExecutor(factory));
            server.start();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void handleRoot(HttpExchange exchange) throws IOException {
        byte[] bytes = renderForm(null).getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().add("Content-Type", "text/html; charset=utf-8");
        exchange.sendResponseHeaders(200, bytes.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(bytes);
        }
    }

    private static void handleConnect(HttpExchange exchange) throws IOException {
        if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
            exchange.sendResponseHeaders(405, -1);
            return;
        }

        String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
        String serverAddr = parseForm(body).get("server");
        if (serverAddr == null || serverAddr.isEmpty()) {
            exchange.sendResponseHeaders(400, -1);
            return;
        }

        MinecraftClient mc = MinecraftClient.getInstance();
        CompletableFuture<byte[]> future = new CompletableFuture<>();
        mc.execute(() -> {
            ConnectScreen.connect(
                new TitleScreen(), mc,
                ServerAddress.parse(serverAddr),
                new ServerInfo("Server", serverAddr, ServerInfo.ServerType.OTHER),
                false, null
            );

            new Thread(() -> {
                try {
                    long start = System.currentTimeMillis();
                    while (true) {
                        if (mc.player != null) break;
                        if (mc.currentScreen instanceof DisconnectedScreen) break;
                        if (System.currentTimeMillis() - start > 20000L) break;
                        Thread.sleep(200);
                    }

                    // Wait a bit longer after the player has loaded before taking
                    // the screenshot. This gives the world time to fully render
                    // before the pause menu is closed and the screenshot is taken.
                    Thread.sleep(3000);

                    mc.execute(() -> {
                        // Temporarily disable pausing on lost focus so the game
                        // doesn't reopen the pause menu before the screenshot.
                        boolean pauseOnLostFocus = mc.options.pauseOnLostFocus;
                        mc.options.pauseOnLostFocus = false;
                        mc.setScreen(null);
                        mc.execute(() -> ScreenshotRecorder.takeScreenshot(mc.getFramebuffer(), image -> {
                            try {
                                Path temp = Files.createTempFile("meteor-web", ".png");
                                image.writeTo(temp);
                                byte[] data = Files.readAllBytes(temp);
                                Files.deleteIfExists(temp);
                                image.close();
                                future.complete(data);
                            } catch (IOException e) {
                                future.completeExceptionally(e);
                            } finally {
                                mc.options.pauseOnLostFocus = pauseOnLostFocus;
                            }
                        }));
                    });
                } catch (InterruptedException e) {
                    future.completeExceptionally(e);
                }
            }, "Meteor-WebCommandServer-Screenshot").start();
        });

        try {
            byte[] bytes = future.get();
            exchange.getResponseHeaders().add("Content-Type", "image/png");
            exchange.sendResponseHeaders(200, bytes.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(bytes);
            }
        } catch (InterruptedException | ExecutionException e) {
            exchange.sendResponseHeaders(500, -1);
        }
    }

    private static Map<String, String> parseForm(String body) {
        Map<String, String> map = new HashMap<>();
        for (String pair : body.split("&")) {
            int idx = pair.indexOf('=');
            if (idx != -1) {
                String key = URLDecoder.decode(pair.substring(0, idx), StandardCharsets.UTF_8);
                String value = URLDecoder.decode(pair.substring(idx + 1), StandardCharsets.UTF_8);
                map.put(key, value);
            }
        }
        return map;
    }

    private static String renderForm(String message) {
        // minimal styles + autofocus; form stays visible after submit
        return "<!DOCTYPE html><html><head><meta charset=\"utf-8\"/>"
            + "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>"
            + "<title>Quick Connect</title>"
            + "<style>"
            + "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;padding:24px;}"
            + "form{display:flex;gap:8px;align-items:center;}"
            + "input[type=text]{padding:8px 10px;font-size:14px;min-width:240px;}"
            + "input[type=submit]{padding:8px 12px;font-size:14px;cursor:pointer;}"
            + ".msg{margin-top:12px;font-size:13px;opacity:.9;}"
            + "</style></head><body>"
            + "<h2>Meteor Quick Connect</h2>"
            + "<form method=\"POST\" action=\"/connect\">"
            + "<input type=\"text\" name=\"server\" placeholder=\"ip:port\" autofocus/>"
            + "<input type=\"submit\" value=\"Connect\"/>"
            + "</form>"
            + (message != null ? "<div class=\"msg\">" + message + "</div>" : "")
            + "</body></html>";
    }

    // very small helper to avoid breaking the message
    private static String escapeHtml(String s) {
        return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;");
    }
}
