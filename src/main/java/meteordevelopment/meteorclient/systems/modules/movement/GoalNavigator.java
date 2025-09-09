package meteordevelopment.meteorclient.systems.modules.movement;

import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.client.network.ClientPlayerEntity;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Locale;

/**
 * Uses a Python reinforcement learning script to move the player
 * towards a goal located 5 blocks away.
 */
public class GoalNavigator extends Module {
    private Process python;
    private BufferedWriter writer;
    private BufferedReader reader;

    private double goalX, goalZ;
    private int ticks;

    public GoalNavigator() {
        super(Categories.Movement, "goal-navigator", "Navigates to a goal using a Python RL model.");
    }

    @Override
    public void onActivate() {
        ClientPlayerEntity player = mc.player;
        if (player == null) return;

        // Goal 5 blocks ahead on X axis
        goalX = player.getX() + 5;
        goalZ = player.getZ();
        ticks = 0;

        try {
            InputStream script = GoalNavigator.class.getResourceAsStream("/scripts/goal_navigator.py");
            if (script == null) {
                error("Script not found");
                toggle();
                return;
            }

            File tmp = File.createTempFile("goal_navigator", ".py");
            tmp.deleteOnExit();
            Files.copy(script, tmp.toPath(), StandardCopyOption.REPLACE_EXISTING);

            ProcessBuilder pb = new ProcessBuilder("python3", "-u", tmp.getAbsolutePath());
            pb.redirectErrorStream(true);
            python = pb.start();
            writer = new BufferedWriter(new OutputStreamWriter(python.getOutputStream()));
            reader = new BufferedReader(new InputStreamReader(python.getInputStream()));
        } catch (IOException e) {
            error("Failed to start python: {}", e.getMessage());
            toggle();
        }
    }

    @Override
    public void onDeactivate() {
        if (mc.player != null) {
            mc.options.forwardKey.setPressed(false);
            mc.options.backKey.setPressed(false);
            mc.options.leftKey.setPressed(false);
            mc.options.rightKey.setPressed(false);
        }
        if (python != null) {
            python.destroy();
            python = null;
        }
    }

    @EventHandler
    private void onTick(TickEvent.Post event) {
        if (python == null || mc.player == null) return;

        double x = mc.player.getX();
        double z = mc.player.getZ();
        double vx = mc.player.getVelocity().x;
        double vz = mc.player.getVelocity().z;

        boolean done = ++ticks >= 20; // 1 second episodes at 20 tps
        if (done) ticks = 0;

        try {
            String line = String.format(Locale.ROOT, "%f,%f,%f,%f,%f,%f,%d\n", x, z, vx, vz, goalX, goalZ, done ? 1 : 0);
            writer.write(line);
            writer.flush();

            String action = reader.readLine();
            if (action != null) applyAction(action.trim());
        } catch (IOException e) {
            error("Python communication error: {}", e.getMessage());
            toggle();
        }
    }

    private void applyAction(String action) {
        mc.options.forwardKey.setPressed("forward".equals(action));
        mc.options.backKey.setPressed("back".equals(action));
        mc.options.leftKey.setPressed("left".equals(action));
        mc.options.rightKey.setPressed("right".equals(action));
    }
}
