package meteordevelopment.meteorclient.systems.modules.misc;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import meteordevelopment.meteorclient.MeteorClient;
import meteordevelopment.meteorclient.events.render.Render3DEvent;
import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.renderer.ShapeMode;
import meteordevelopment.meteorclient.settings.*;
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
 * Multi-head DQN navigator with composite actions:
 * move, strafe, yaw, pitch, jump can be applied in the same tick.
 */
public class QPathFinder extends Module {
    private static final double MIN_GOAL_DISTANCE = 3.0;
    private static final double MAX_GOAL_SPAWN_RANGE = 20.0;
    private static final double START_GOAL_SPAWN_RANGE = 8.0;

    private static final double YAW_STEP_DEG = 7.5;
    private static final double PITCH_STEP_DEG = 5.0;

    private static final int STUCK_TICKS = 30;

    private final SettingGroup sgGeneral = settings.getDefaultGroup();
    private final Setting<Boolean> evaluation = sgGeneral.add(new BoolSetting.Builder()
        .name("evaluation")
        .description("Runs the model without training.")
        .defaultValue(false)
        .build()
    );

    private Process python;
    private BufferedWriter writer;
    private BufferedReader reader;

    private double goalX, goalY, goalZ;
    private double startX, startZ;
    private int step;
    private int episodeMaxSteps;
    private int noProgressTicks;
    private double lastDist;

    private long startTime;
    private long totalTime;
    private int goalsReached;

    private double goalSpawnRange = START_GOAL_SPAWN_RANGE;

    public QPathFinder() {
        super(Categories.Misc, "q-path-finder", "Trains a DQN agent to navigate to a goal with composite actions.");
    }

    @Override
    public void onActivate() {
        try {
            InputStream script = QPathFinder.class.getResourceAsStream("/scripts/q_path_finder.py");
            if (script == null) {
                error("Could not find Python script.");
                toggle();
                return;
            }

            File tmp = File.createTempFile("q_path_finder", ".py");
            tmp.deleteOnExit();
            Files.copy(script, tmp.toPath(), StandardCopyOption.REPLACE_EXISTING);

            File checkpoint = new File(MeteorClient.FOLDER, "q_path_finder_checkpoint.pth");
            if (checkpoint.exists()) info("Loading checkpoint from %s", checkpoint.getAbsolutePath());
            else info("No checkpoint found, starting fresh.");

            python = new ProcessBuilder("python", tmp.getAbsolutePath(), checkpoint.getAbsolutePath(), evaluation.get().toString()).start();
            writer = new BufferedWriter(new OutputStreamWriter(python.getOutputStream()));
            reader = new BufferedReader(new InputStreamReader(python.getInputStream()));

            // Log stderr from python
            new Thread(() -> {
                try (BufferedReader err = new BufferedReader(new InputStreamReader(python.getErrorStream()))) {
                    String line;
                    while ((line = err.readLine()) != null) error("PY: %s", line);
                } catch (IOException ignored) {}
            }, "q-path-finder-py").start();

            totalTime = 0;
            goalsReached = 0;
            resetGoal();
            info(evaluation.get() ? "Evaluation started." : "Training started.");
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
        mc.options.leftKey.setPressed(false);   // strafe left
        mc.options.rightKey.setPressed(false);  // strafe right
        mc.options.jumpKey.setPressed(false);
    }

    // Composite action application.
    // move: 0 idle, 1 forward, 2 back
    // strafe: 0 idle, 1 left, 2 right
    // yaw: 0 none, 1 left, 2 right
    // pitch: 0 none, 1 up, 2 down
    // jump: 0 no, 1 yes
    private void applyCompositeAction(int move, int strafe, int yaw, int pitch, int jump) {
        unpress();

        // Movement keys (can co-exist)
        if (move == 1) mc.options.forwardKey.setPressed(true);
        else if (move == 2) mc.options.backKey.setPressed(true);

        if (strafe == 1) mc.options.leftKey.setPressed(true);
        else if (strafe == 2) mc.options.rightKey.setPressed(true);

        if (jump == 1) mc.options.jumpKey.setPressed(true);

        // View control: small continuous adjustments
        float curYaw = mc.player.getYaw();
        float curPitch = mc.player.getPitch();

        if (yaw == 1) curYaw -= (float) YAW_STEP_DEG;
        else if (yaw == 2) curYaw += (float) YAW_STEP_DEG;

        if (pitch == 1) curPitch -= (float) PITCH_STEP_DEG;   // look up
        else if (pitch == 2) curPitch += (float) PITCH_STEP_DEG; // look down

        // Clamp pitch to [-90, 90]
        if (curPitch < -90f) curPitch = -90f;
        if (curPitch > 90f) curPitch = 90f;

        mc.player.setYaw(curYaw);
        mc.player.setPitch(curPitch);
    }

    private void resetGoal() {
        startX = mc.player.getX();
        startZ = mc.player.getZ();

        double dx, dz;
        do {
            dx = ThreadLocalRandom.current().nextDouble(-goalSpawnRange, goalSpawnRange);
            dz = ThreadLocalRandom.current().nextDouble(-goalSpawnRange, goalSpawnRange);
        } while ((dx * dx + dz * dz) < (MIN_GOAL_DISTANCE * MIN_GOAL_DISTANCE));

        goalX = startX + dx;
        goalZ = startZ + dz;
        goalY = mc.player.getY();

        double initialDistance = Math.hypot(dx, dz);

        // Allocate time based on distance
        episodeMaxSteps = Math.min(600, Math.max(120, (int) Math.round(initialDistance * 6 + 40)));

        step = 0;
        noProgressTicks = 0;
        lastDist = initialDistance;
        startTime = System.currentTimeMillis();

        info("New goal X: %.1f Y: %.1f Z: %.1f | dist=%.1f | steps=%d | range=%.1f",
            goalX, goalY, goalZ, initialDistance, episodeMaxSteps, goalSpawnRange);
    }

    @EventHandler
    private void onTick(TickEvent.Pre event) {
        if (python == null || mc.player == null) return;

        double px = mc.player.getX();
        double pz = mc.player.getZ();
        double vx = mc.player.getVelocity().x;
        double vz = mc.player.getVelocity().z;
        double yaw = mc.player.getYaw();

        double pxRel = px - startX;
        double pzRel = pz - startZ;
        double gxRel = goalX - startX;
        double gzRel = goalZ - startZ;

        // World distance
        double dist = Math.hypot(gxRel - pxRel, gzRel - pzRel);
        double progress = lastDist - dist;
        if (Math.abs(progress) < 0.02) noProgressTicks++;
        else noProgressTicks = 0;
        lastDist = dist;

        boolean reached = dist < 1.5;
        boolean timeout = step++ >= episodeMaxSteps;
        boolean stuck = noProgressTicks >= STUCK_TICKS;
        boolean done = reached || timeout || stuck;

        JsonObject obj = new JsonObject();
        obj.addProperty("px", pxRel);
        obj.addProperty("pz", pzRel);
        obj.addProperty("vx", vx);
        obj.addProperty("vz", vz);
        obj.addProperty("yaw", yaw);
        obj.addProperty("gx", gxRel);
        obj.addProperty("gz", gzRel);
        obj.addProperty("dist", dist);
        obj.addProperty("steps_left", Math.max(0, episodeMaxSteps - step));
        obj.addProperty("done", done);
        obj.addProperty("reached", reached);
        obj.addProperty("stuck", stuck);

        try {
            writer.write(obj.toString());
            writer.newLine();
            writer.flush();

            String line = reader.readLine();
            if (line != null && !line.isEmpty()) {
                // Expect JSON composite action
                try {
                    JsonObject act = JsonParser.parseString(line.trim()).getAsJsonObject();
                    int move   = act.has("move")  ? act.get("move").getAsInt()  : 0;
                    int strafe = act.has("strafe")? act.get("strafe").getAsInt(): 0;
                    int yawA   = act.has("yaw")   ? act.get("yaw").getAsInt()   : 0;
                    int pitchA = act.has("pitch") ? act.get("pitch").getAsInt() : 0;
                    int jump   = act.has("jump")  ? act.get("jump").getAsInt()  : 0;
                    applyCompositeAction(move, strafe, yawA, pitchA, jump);
                } catch (Exception parseErr) {
                    // Backward-compat: if it's a single int, just idle.
                    applyCompositeAction(0, 0, 0, 0, 0);
                }
            }

            if (done) {
                if (reached) {
                    long elapsed = System.currentTimeMillis() - startTime;
                    totalTime += elapsed;
                    goalsReached++;
                    double avg = totalTime / (double) goalsReached / 1000.0;
                    info("Goal reached in %d steps (%.2fs). Avg time: %.2fs", step, elapsed / 1000.0, avg);
                    goalSpawnRange = Math.min(MAX_GOAL_SPAWN_RANGE, goalSpawnRange + 1.0);
                } else if (stuck) {
                    info("Episode ended due to lack of progress.");
                    goalSpawnRange = Math.max(6.0, goalSpawnRange - 0.5);
                } else {
                    info("Episode timed out.");
                    goalSpawnRange = Math.max(6.0, goalSpawnRange - 0.5);
                }
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
