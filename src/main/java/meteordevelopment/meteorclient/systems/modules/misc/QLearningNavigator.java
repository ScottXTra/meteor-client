package meteordevelopment.meteorclient.systems.modules.misc;

import com.google.gson.JsonObject;
import meteordevelopment.meteorclient.MeteorClient;
import meteordevelopment.meteorclient.events.render.Render3DEvent;
import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.renderer.ShapeMode;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.orbit.EventHandler;
import meteordevelopment.meteorclient.utils.render.color.Color;
import net.minecraft.util.math.Box;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Module that communicates with a Python Q-learning script to train a bot to reach random goals.
 * When enabled, player state and target coordinates are streamed to the Python process which
 * responds with movement actions. Debug information is printed to the Minecraft chat.
 */
public class QLearningNavigator extends Module {
    private static final int MAX_STEPS = 200;

    private Process python;
    private BufferedWriter writer;
    private BufferedReader reader;

    private double goalX, goalY, goalZ;
    private int step;

    private long startTime;
    private long totalTime;
    private int goalsReached;

    public QLearningNavigator() {
        super(Categories.Misc, "qlearning-navigator", "Trains a Q-learning agent to navigate to a goal.");
    }

    @Override
    public void onActivate() {
        try {
            InputStream script = QLearningNavigator.class.getResourceAsStream("/scripts/ql_nav.py");
            if (script == null) {
                error("Could not find Python script.");
                toggle();
                return;
            }

            File tmp = File.createTempFile("ql_nav", ".py");
            tmp.deleteOnExit();
            Files.copy(script, tmp.toPath(), StandardCopyOption.REPLACE_EXISTING);

            File checkpoint = new File(MeteorClient.FOLDER, "ql_nav_checkpoint.pth");
            if (checkpoint.exists()) info("Loading checkpoint from %s", checkpoint.getAbsolutePath());
            else info("No checkpoint found, starting fresh.");
            python = new ProcessBuilder("python", tmp.getAbsolutePath(), checkpoint.getAbsolutePath()).start();
            writer = new BufferedWriter(new OutputStreamWriter(python.getOutputStream()));
            reader = new BufferedReader(new InputStreamReader(python.getInputStream()));

            // Log stderr from python
            new Thread(() -> {
                try (BufferedReader err = new BufferedReader(new InputStreamReader(python.getErrorStream()))) {
                    String line;
                    while ((line = err.readLine()) != null) {
                        error("PY: %s", line);
                    }
                } catch (IOException ignored) {
                }
            }, "ql-nav-py").start();

            totalTime = 0;
            goalsReached = 0;
            resetGoal();
            info("Training started.");
        } catch (IOException e) {
            error("Python start failed: %s", e.getMessage());
            toggle();
        }
    }

    @Override
    public void onDeactivate() {
        stopPython();
        unpress();
    }

    private void stopPython() {
        if (python != null) {
            python.destroy();
            python = null;
        }
    }

    private void unpress() {
        mc.options.forwardKey.setPressed(false);
        mc.options.backKey.setPressed(false);
        mc.options.leftKey.setPressed(false);
        mc.options.rightKey.setPressed(false);
    }

    private void applyAction(int action) {
        unpress();
        switch (action) {
            case 0 -> mc.options.forwardKey.setPressed(true);
            case 1 -> mc.options.backKey.setPressed(true);
            case 2 -> mc.options.leftKey.setPressed(true);
            case 3 -> mc.options.rightKey.setPressed(true);
            default -> {
            }
        }
    }

    private void resetGoal() {
        goalX = mc.player.getX() + ThreadLocalRandom.current().nextDouble(-20, 20);
        goalZ = mc.player.getZ() + ThreadLocalRandom.current().nextDouble(-20, 20);
        goalY = mc.player.getY();
        step = 0;
        startTime = System.currentTimeMillis();
        info("New goal X: %.1f Y: %.1f Z: %.1f", goalX, goalY, goalZ);
    }

    @EventHandler
    private void onTick(TickEvent.Pre event) {
        if (python == null || mc.player == null) return;

        double px = mc.player.getX();
        double pz = mc.player.getZ();
        double vx = mc.player.getVelocity().x;
        double vz = mc.player.getVelocity().z;

        double dist = Math.hypot(goalX - px, goalZ - pz);
        boolean reached = dist < 1.5;
        boolean timeout = step++ >= MAX_STEPS;
        boolean done = reached || timeout;

        JsonObject obj = new JsonObject();
        obj.addProperty("px", px);
        obj.addProperty("pz", pz);
        obj.addProperty("vx", vx);
        obj.addProperty("vz", vz);
        obj.addProperty("gx", goalX);
        obj.addProperty("gz", goalZ);
        obj.addProperty("done", done);
        obj.addProperty("reached", reached);

        try {
            writer.write(obj.toString());
            writer.newLine();
            writer.flush();

            String line = reader.readLine();
            if (line != null && !line.isEmpty()) {
                int action = Integer.parseInt(line.trim());
                applyAction(action);
            }

            if (done) {
                if (reached) {
                    long elapsed = System.currentTimeMillis() - startTime;
                    totalTime += elapsed;
                    goalsReached++;
                    double avg = totalTime / (double) goalsReached / 1000.0;
                    info("Goal reached in %d steps (%.2fs). Avg time: %.2fs", step, elapsed / 1000.0, avg);
                } else info("Episode timed out.");
                resetGoal();
            }
        } catch (Exception e) {
            error("Python error: %s", e.getMessage());
            toggle();
        }
    }

    @EventHandler
    private void onRender(Render3DEvent event) {
        if (mc.player == null) return;
        Box box = new Box(goalX - 0.5, goalY, goalZ - 0.5, goalX + 0.5, goalY + 1, goalZ + 0.5);
        event.renderer.box(box, Color.GREEN, Color.GREEN, ShapeMode.Lines, 0);
    }
}
