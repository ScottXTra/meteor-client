package meteordevelopment.meteorclient.systems.modules.misc;

import com.google.gson.JsonObject;
import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.orbit.EventHandler;

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

    private double goalX, goalZ;
    private double prevDist;
    private int step;

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

            python = new ProcessBuilder("python", tmp.getAbsolutePath()).start();
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
        prevDist = Double.MAX_VALUE;
        step = 0;
        info("New goal X: %.1f Z: %.1f", goalX, goalZ);
    }

    @EventHandler
    private void onTick(TickEvent.Pre event) {
        if (python == null || mc.player == null) return;

        double px = mc.player.getX();
        double pz = mc.player.getZ();
        double vx = mc.player.getVelocity().x;
        double vz = mc.player.getVelocity().z;

        double dist = Math.hypot(goalX - px, goalZ - pz);
        double reward = (prevDist == Double.MAX_VALUE ? 0 : prevDist - dist) - 0.01;
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
        obj.addProperty("reward", reward);
        obj.addProperty("done", done);

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
                if (reached) info("Goal reached in %d steps.", step);
                else info("Episode timed out.");
                resetGoal();
            }
        } catch (Exception e) {
            error("Python error: %s", e.getMessage());
            toggle();
        }

        prevDist = dist;
    }
}
