// CLIENT MODULE (Java) — drop-in replacement

package meteordevelopment.meteorclient.systems.modules.movement;

import meteordevelopment.meteorclient.MeteorClient;
import meteordevelopment.meteorclient.events.render.Render3DEvent;
import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.renderer.ShapeMode;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.utils.render.RenderUtils;
import meteordevelopment.meteorclient.utils.render.color.Color;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.Vec3d;

import java.io.*;
import java.util.concurrent.TimeUnit;

public class PythonQLearning extends Module {
    private Process process;
    private BufferedWriter writer;
    private BufferedReader reader;
    private BufferedReader errReader;
    private Thread debugThread;

    // Episode framing
    private Vec3d episodeStart;      // start position for the current episode
    private Vec3d goal;              // absolute goal (y = -60)
    private Vec3d goalRel;           // goal relative to episodeStart in XZ
    private boolean goalChanged;
    private boolean goalFailed;
    private boolean goalSucceeded;
    private int ticks;
    private int timeoutTicksForGoal;

    // Action state
    private static final String DEFAULT_ACTION = "none";
    private String lastAction = DEFAULT_ACTION;

    // Navigation config
    private static final double MAX_GOAL_DIST = 200.0;
    private static final double REACH_THRESHOLD = 0.75; // horizontal (XZ) meters
    private static final int MIN_TIMEOUT_TICKS = 30;    // 1.5s @ 20 tps
    private static final int MAX_TIMEOUT_TICKS = 200;   // 10s @ 20 tps
    private static final float TURN_DEG_PER_TICK = 7.5f;
    private static final double GOAL_SAMPLE_RADIUS = 4.0; // small local hops

    private static final double FIXED_GOAL_Y = -60.0;

    private File checkpointFile;

    public PythonQLearning() {
        super(Categories.Movement, "python-qlearning", "Moves the player using a Python DQN agent.");
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

            checkpointFile = new File(MeteorClient.FOLDER, "python_qlearning_checkpoint.pt");

            process = new ProcessBuilder("python", temp.getAbsolutePath(), checkpointFile.getAbsolutePath())
                .redirectErrorStream(false)
                .start();

            writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()));
            reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            errReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
        } catch (IOException e) {
            error("Failed to start Python: {}", e.getMessage());
            toggle();
            return;
        }

        // initial episode start
        if (mc.player != null) episodeStart = mc.player.getPos();

        // forward stderr from Python to chat for visibility
        debugThread = new Thread(this::readDebug, "python-qlearning-debug");
        debugThread.start();

        // clear state
        goal = null;
        goalRel = null;
        goalChanged = false;
        goalFailed = false;
        goalSucceeded = false;
        lastAction = DEFAULT_ACTION;
        ticks = 0;
    }

    @Override
    public void onDeactivate() {
        // release all keys
        mc.options.forwardKey.setPressed(false);
        mc.options.backKey.setPressed(false);
        mc.options.leftKey.setPressed(false);
        mc.options.rightKey.setPressed(false);
        mc.options.jumpKey.setPressed(false);

        // ask backend to save
        try {
            if (writer != null) {
                writer.write("{\"save\":true}\n");
                writer.flush();
            }
        } catch (IOException ignored) {}

        try {
            if (writer != null) writer.close();
            if (reader != null) reader.close();
            if (errReader != null) errReader.close();
        } catch (IOException ignored) {}

        if (debugThread != null) {
            debugThread.interrupt();
            debugThread = null;
        }

        if (process != null) {
            try {
                process.waitFor(1, TimeUnit.SECONDS);
            } catch (InterruptedException ignored) {}
            process.destroyForcibly();
            process = null;
        }

        goal = null;
        goalRel = null;
        goalChanged = false;
        goalFailed = false;
        goalSucceeded = false;
        episodeStart = null;
        lastAction = DEFAULT_ACTION;
        ticks = 0;
    }

    // --------------------------------------
    // Tick-driven I/O: one observation+action per game tick
    // --------------------------------------
    @EventHandler
    private void onTick(TickEvent.Post event) {
        if (!isActive() || mc.player == null || mc.world == null) return;
        if (writer == null || reader == null) return;

        try {
            if (goal == null) setNewEpisodeAndGoal();

            // Compose one observation per tick
            Vec3d p = mc.player.getPos();
            float yaw = mc.player.getYaw();
            float pitch = mc.player.getPitch();

            String msg = String.format(
                "{\"player\":{\"x\":%f,\"y\":%f,\"z\":%f,\"yaw\":%f,\"pitch\":%f},"
                    + "\"start\":{\"x\":%f,\"y\":%f,\"z\":%f},"
                    + "\"goal\":{\"x\":%f,\"y\":%f,\"z\":%f},"
                    + "\"goal_rel\":{\"dx\":%f,\"dz\":%f}%s%s%s}\n",
                p.x, p.y, p.z, yaw, pitch,
                episodeStart.x, episodeStart.y, episodeStart.z,
                goal.x, goal.y, goal.z,
                goalRel.x, goalRel.z,
                goalChanged ? ",\"reset\":true" : "",
                goalFailed ? ",\"fail\":true" : "",
                goalSucceeded ? ",\"success\":true" : ""
            );

            // Clear flags after sending
            goalChanged = false;
            goalFailed = false;
            goalSucceeded = false;

            writer.write(msg);
            writer.flush();

            // Non-blocking read: consume the latest available action for this tick
            String actionStr = null;
            while (reader.ready()) {
                String line = reader.readLine();
                if (line == null) break;
                actionStr = line.trim();
            }
            if (actionStr != null && !actionStr.isEmpty()) {
                lastAction = actionStr;
            }

            // Apply exactly once per tick
            applyAction(lastAction);

            // Success/timeout bookkeeping (horizontal distance only)
            Vec3d p2 = mc.player.getPos();
            double dist2 = horizontalDistSq(p2, goal);
            if (dist2 < REACH_THRESHOLD * REACH_THRESHOLD) {
                goalSucceeded = true;
                setNewEpisodeAndGoal();
            } else {
                ticks++;
                if (ticks >= timeoutTicksForGoal) {
                    goalFailed = true;
                    setNewEpisodeAndGoal();
                }
            }
        } catch (Exception e) {
            error("Python I/O error: {}", e.getMessage());
            toggle();
        }
    }

    private double horizontalDistSq(Vec3d a, Vec3d b) {
        double dx = a.x - b.x;
        double dz = a.z - b.z;
        return dx * dx + dz * dz;
    }

    private void applyAction(String action) {
        // release all keys first
        mc.options.forwardKey.setPressed(false);
        mc.options.backKey.setPressed(false);
        mc.options.leftKey.setPressed(false);
        mc.options.rightKey.setPressed(false);
        mc.options.jumpKey.setPressed(false);

        switch (action) {
            case "forward" -> mc.options.forwardKey.setPressed(true);
            case "back" -> mc.options.backKey.setPressed(true);
            case "left" -> mc.options.leftKey.setPressed(true);
            case "right" -> mc.options.rightKey.setPressed(true);
            case "jump" -> mc.options.jumpKey.setPressed(true);
            case "turn_left" -> mc.player.setYaw(mc.player.getYaw() + TURN_DEG_PER_TICK);
            case "turn_right" -> mc.player.setYaw(mc.player.getYaw() - TURN_DEG_PER_TICK);
            case "none" -> { /* do nothing */ }
            default -> { /* unknown action: ignore */ }
        }
    }

    private void setNewEpisodeAndGoal() {
        if (mc.player == null) return;

        // New episode starts from current player position
        episodeStart = mc.player.getPos();

        // Sample a small relative target in XZ (within GOAL_SAMPLE_RADIUS), then clamp to MAX_GOAL_DIST
        double angle = Math.random() * Math.PI * 2.0;
        double dx = GOAL_SAMPLE_RADIUS * Math.cos(angle);
        double dz = GOAL_SAMPLE_RADIUS * Math.sin(angle);

        Vec3d rel = new Vec3d(dx, 0.0, dz);
        double relLenSq = rel.x * rel.x + rel.z * rel.z;
        if (relLenSq > MAX_GOAL_DIST * MAX_GOAL_DIST) {
            double len = Math.sqrt(relLenSq);
            rel = new Vec3d(rel.x / len * MAX_GOAL_DIST, 0.0, rel.z / len * MAX_GOAL_DIST);
        }

        goalRel = rel;
        goal = new Vec3d(episodeStart.x + rel.x, FIXED_GOAL_Y, episodeStart.z + rel.z);

        goalChanged = true;
        ticks = 0;

        // timeout scales with horizontal distance from start to goal
        setAdaptiveTimeout(goal, episodeStart);
    }

    private void setAdaptiveTimeout(Vec3d g, Vec3d p) {
        double d = Math.max(1.0, Math.hypot(g.x - p.x, g.z - p.z)); // horizontal blocks
        // ~6 ticks per block, clamped
        timeoutTicksForGoal = MathHelper.clamp((int) Math.round(d * 6.0), MIN_TIMEOUT_TICKS, MAX_TIMEOUT_TICKS);
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
                final String msg = line;
                mc.execute(() -> info(msg));
            }
        } catch (IOException ignored) {}
    }
}
