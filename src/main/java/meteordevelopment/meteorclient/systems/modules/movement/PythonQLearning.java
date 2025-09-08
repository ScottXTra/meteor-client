package meteordevelopment.meteorclient.systems.modules.movement;

import meteordevelopment.meteorclient.events.render.Render3DEvent;
import meteordevelopment.meteorclient.renderer.ShapeMode;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.utils.render.RenderUtils;
import meteordevelopment.meteorclient.utils.render.color.Color;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.Heightmap;

import java.io.*;

public class PythonQLearning extends Module {
    private Process process;
    private BufferedWriter writer;
    private BufferedReader reader;
    private BufferedReader errReader;
    private Thread thread;
    private Thread debugThread;
    private Vec3d goal;
    private boolean goalChanged;
    private Vec3d startPos;
    private static final double MAX_GOAL_DIST = 200;

    public PythonQLearning() {
        super(Categories.Movement, "python-qlearning", "Moves the player using a Python Q-learning agent.");
    }

    @Override
    public void onActivate() {
        try {
            InputStream script = PythonQLearning.class.getResourceAsStream("/scripts/qlearning_agent.py");
            if (script == null) {
                error("Could not find Python script.");
                toggle();
                return;
            }
            File temp = File.createTempFile("qlearning_agent", ".py");
            try (FileOutputStream out = new FileOutputStream(temp)) {
                script.transferTo(out);
            }
            temp.deleteOnExit();
            process = new ProcessBuilder("python", temp.getAbsolutePath()).start();
            writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()));
            reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            errReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
        } catch (IOException e) {
            error("Failed to start Python: {}", e.getMessage());
            toggle();
            return;
        }

        if (mc.player != null) startPos = mc.player.getPos();

        thread = new Thread(this::loop, "python-qlearning-loop");
        thread.start();

        debugThread = new Thread(this::readDebug, "python-qlearning-debug");
        debugThread.start();
    }

    private void loop() {
        try {
            while (isActive() && mc.player != null && !Thread.interrupted()) {
                if (goal == null) setGoal();

                Vec3d p = mc.player.getPos();
                String msg = String.format("{\"player\":{\"x\":%f,\"y\":%f,\"z\":%f,\"yaw\":%f,\"pitch\":%f},\"goal\":{\"x\":%f,\"y\":%f,\"z\":%f}%s}\n",
                    p.x, p.y, p.z, mc.player.getYaw(), mc.player.getPitch(), goal.x, goal.y, goal.z, goalChanged ? ",\"reset\":true" : "");
                goalChanged = false;
                writer.write(msg);
                writer.flush();

                String action = reader.readLine();
                if (action == null) break;
                mc.execute(() -> applyAction(action.trim()));
                Thread.sleep(50);

                Vec3d p2 = mc.player.getPos();
                if (p2.squaredDistanceTo(goal) < 1) setGoal();
            }
        } catch (Exception e) {
            error("Python communication error: {}", e.getMessage());
        }
        mc.execute(this::toggle);
    }

    private void applyAction(String action) {
        mc.options.forwardKey.setPressed(action.equals("forward") || action.equals("sprint-forward"));
        mc.options.backKey.setPressed(action.equals("back"));
        mc.options.leftKey.setPressed(action.equals("left"));
        mc.options.rightKey.setPressed(action.equals("right"));
        mc.options.sprintKey.setPressed(action.equals("sprint-forward"));

        float yaw = mc.player.getYaw();
        float pitch = mc.player.getPitch();

        if (action.equals("look-left")) mc.player.setYaw(yaw - 5);
        else if (action.equals("look-right")) mc.player.setYaw(yaw + 5);
        else if (action.equals("look-up")) mc.player.setPitch(MathHelper.clamp(pitch - 5, -90, 90));
        else if (action.equals("look-down")) mc.player.setPitch(MathHelper.clamp(pitch + 5, -90, 90));
    }

    @Override
    public void onDeactivate() {
        mc.options.forwardKey.setPressed(false);
        mc.options.backKey.setPressed(false);
        mc.options.leftKey.setPressed(false);
        mc.options.rightKey.setPressed(false);
        mc.options.sprintKey.setPressed(false);

        if (thread != null) {
            thread.interrupt();
            thread = null;
        }
        if (debugThread != null) {
            debugThread.interrupt();
            debugThread = null;
        }

        try {
            if (writer != null) writer.close();
            if (reader != null) reader.close();
            if (errReader != null) errReader.close();
        } catch (IOException ignored) {
        }

        if (process != null) {
            process.destroyForcibly();
            process = null;
        }

        goal = null;
        goalChanged = false;
        startPos = null;
    }

    private void setGoal() {
        if (mc.player == null || mc.world == null) return;

        Vec3d p = mc.player.getPos();
        int playerY = mc.player.getBlockY();

        // Try to find a random position at the same Y level as the player
        for (int i = 0; i < 50; i++) {
            double angle = Math.random() * Math.PI * 2;
            int x = MathHelper.floor(p.x + 12 * Math.cos(angle));
            int z = MathHelper.floor(p.z + 12 * Math.sin(angle));

            int topY = mc.world.getTopY(Heightmap.Type.MOTION_BLOCKING, x, z);
            if (topY == playerY) {
                Vec3d newGoal = new Vec3d(x + 0.5, playerY, z + 0.5);
                if (startPos != null && newGoal.squaredDistanceTo(startPos) > MAX_GOAL_DIST * MAX_GOAL_DIST) {
                    Vec3d dir = newGoal.subtract(startPos).normalize().multiply(MAX_GOAL_DIST);
                    newGoal = startPos.add(dir);
                }
                goal = newGoal;
                goalChanged = true;
                return;
            }
        }

        // Fallback if no matching Y level was found
        double angle = Math.random() * Math.PI * 2;
        Vec3d newGoal = new Vec3d(p.x + 12 * Math.cos(angle), playerY, p.z + 12 * Math.sin(angle));
        if (startPos != null && newGoal.squaredDistanceTo(startPos) > MAX_GOAL_DIST * MAX_GOAL_DIST) {
            Vec3d dir = newGoal.subtract(startPos).normalize().multiply(MAX_GOAL_DIST);
            newGoal = startPos.add(dir);
        }
        goal = newGoal;
        goalChanged = true;
    }

    @EventHandler
    private void onRender(Render3DEvent event) {
        if (goal == null) return;
        event.renderer.box(goal.x - 0.5, goal.y, goal.z - 0.5, goal.x + 0.5, goal.y + 2, goal.z + 0.5, Color.RED, Color.RED, ShapeMode.Lines, 0);
        event.renderer.line(RenderUtils.center.x, RenderUtils.center.y, RenderUtils.center.z, goal.x, goal.y, goal.z, Color.GREEN);
    }

    private void readDebug() {
        try {
            String line;
            while ((line = errReader.readLine()) != null && !Thread.interrupted()) {
                String msg = line;
                mc.execute(() -> info(msg));
            }
        } catch (IOException ignored) {
        }
    }
}
