package meteordevelopment.meteorclient.systems.modules.misc;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import meteordevelopment.meteorclient.MeteorClient;
import meteordevelopment.meteorclient.events.render.Render3DEvent;
import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.gui.GuiTheme;
import meteordevelopment.meteorclient.gui.widgets.WWidget;
import meteordevelopment.meteorclient.gui.widgets.containers.WVerticalList;
import meteordevelopment.meteorclient.gui.widgets.pressable.WButton;
import meteordevelopment.meteorclient.renderer.ShapeMode;
import meteordevelopment.meteorclient.settings.BlockPosSetting;
import meteordevelopment.meteorclient.settings.BoolSetting;
import meteordevelopment.meteorclient.settings.DoubleSetting;
import meteordevelopment.meteorclient.settings.Setting;
import meteordevelopment.meteorclient.settings.SettingGroup;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.utils.player.ChatUtils;
import meteordevelopment.meteorclient.utils.render.color.Color;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Box;
import net.minecraft.util.math.Vec3d;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

public class ActivityTrainer extends Module {
    private static final int WAIT_TICKS = 40;
    private static final int MAX_RUN_TICKS = 160;
    private static final double YAW_STEP_DEG = 7.5;
    private static final double PITCH_STEP_DEG = 5.0;

    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private final Setting<Boolean> evaluation = sgGeneral.add(new BoolSetting.Builder()
        .name("evaluation")
        .description("Runs the model without training.")
        .defaultValue(false)
        .build()
    );

    private final Setting<BlockPos> startPoint = sgGeneral.add(new BlockPosSetting.Builder()
        .name("start-point")
        .description("Block used as the beginning of each training episode.")
        .defaultValue(BlockPos.ORIGIN)
        .build()
    );

    private final Setting<BlockPos> goalPoint = sgGeneral.add(new BlockPosSetting.Builder()
        .name("goal-point")
        .description("Block used as the target for each training episode.")
        .defaultValue(BlockPos.ORIGIN)
        .build()
    );

    private final Setting<Double> goalRadius = sgGeneral.add(new DoubleSetting.Builder()
        .name("goal-radius")
        .description("Distance in blocks considered a successful goal reach.")
        .defaultValue(0.9)
        .min(0.25)
        .sliderRange(0.25, 5.0)
        .build()
    );

    private Process python;
    private BufferedWriter writer;
    private BufferedReader reader;

    private Thread stderrThread;

    private BlockPos cachedStart;
    private BlockPos cachedGoal;
    private Vec3d startVec;
    private Vec3d goalVec;
    private String teleportCommand;

    private EpisodePhase phase = EpisodePhase.WAITING;
    private int waitTicks;
    private int runTicks;
    private double lastDistance;
    private boolean pendingReset;

    private long episodeCounter;
    private long successes;

    public ActivityTrainer() {
        super(Categories.Misc, "activity-trainer", "Trains a Python-controlled agent to reach a goal from a fixed start.");
    }

    @Override
    public void onActivate() {
        if (mc.player == null) {
            error("Player not available.");
            toggle();
            return;
        }

        refreshTargets();
        if (startVec == null || goalVec == null) {
            error("Set both start and goal points before activating.");
            toggle();
            return;
        }

        if (cachedStart.equals(cachedGoal)) {
            error("Start and goal points must be different.");
            toggle();
            return;
        }

        try {
            InputStream script = ActivityTrainer.class.getResourceAsStream("/scripts/activity_trainer.py");
            if (script == null) {
                error("Could not find Python script.");
                toggle();
                return;
            }

            File tmp = File.createTempFile("activity_trainer", ".py");
            tmp.deleteOnExit();
            Files.copy(script, tmp.toPath(), StandardCopyOption.REPLACE_EXISTING);

            File checkpoint = new File(MeteorClient.FOLDER, "activity_trainer_checkpoint.pth");
            if (checkpoint.exists()) info("Loading checkpoint from %s", checkpoint.getAbsolutePath());
            else info("No checkpoint found, starting fresh.");

            python = new ProcessBuilder("python", tmp.getAbsolutePath(), checkpoint.getAbsolutePath(), evaluation.get().toString()).start();
            writer = new BufferedWriter(new OutputStreamWriter(python.getOutputStream()));
            reader = new BufferedReader(new InputStreamReader(python.getInputStream()));

            stderrThread = new Thread(() -> {
                try (BufferedReader err = new BufferedReader(new InputStreamReader(python.getErrorStream()))) {
                    String line;
                    while ((line = err.readLine()) != null) error("PY: %s", line);
                } catch (IOException ignored) {}
            }, "activity-trainer-py");
            stderrThread.setDaemon(true);
            stderrThread.start();

            episodeCounter = 0;
            successes = 0;

            beginWaitingPhase(true);
            teleportToStart();

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
        phase = EpisodePhase.WAITING;
    }

    private void stopPython() {
        if (python != null) {
            python.destroy();
            python = null;
        }
        writer = null;
        reader = null;
        stderrThread = null;
    }

    private void unpress() {
        mc.options.forwardKey.setPressed(false);
        mc.options.backKey.setPressed(false);
        mc.options.leftKey.setPressed(false);
        mc.options.rightKey.setPressed(false);
        mc.options.jumpKey.setPressed(false);
        mc.options.sprintKey.setPressed(false);
    }

    private void applyCompositeAction(int move, int strafe, int yaw, int pitch, int jump) {
        unpress();

        if (move == 1) mc.options.forwardKey.setPressed(true);
        else if (move == 2) mc.options.backKey.setPressed(true);

        if (strafe == 1) mc.options.leftKey.setPressed(true);
        else if (strafe == 2) mc.options.rightKey.setPressed(true);

        if (jump == 1) mc.options.jumpKey.setPressed(true);

        if (mc.player == null) return;

        float curYaw = mc.player.getYaw();
        float curPitch = mc.player.getPitch();

        if (yaw == 1) curYaw -= (float) YAW_STEP_DEG;
        else if (yaw == 2) curYaw += (float) YAW_STEP_DEG;

        if (pitch == 1) curPitch -= (float) PITCH_STEP_DEG;
        else if (pitch == 2) curPitch += (float) PITCH_STEP_DEG;

        if (curPitch < -90f) curPitch = -90f;
        if (curPitch > 90f) curPitch = 90f;

        mc.player.setYaw(curYaw);
        mc.player.setPitch(curPitch);
    }

    private void refreshTargets() {
        BlockPos start = startPoint.get();
        BlockPos goal = goalPoint.get();

        if (cachedStart == null || !cachedStart.equals(start)) {
            cachedStart = start;
            startVec = topCenter(start);
            teleportCommand = String.format("/tp @p %.2f %.2f %.2f", startVec.x, startVec.y, startVec.z);
        }

        if (cachedGoal == null || !cachedGoal.equals(goal)) {
            cachedGoal = goal;
            goalVec = topCenter(goal);
        }
    }

    private Vec3d topCenter(BlockPos pos) {
        return new Vec3d(pos.getX() + 0.5, pos.getY() + 1.0, pos.getZ() + 0.5);
    }

    private void beginWaitingPhase(boolean initial) {
        phase = EpisodePhase.WAITING;
        waitTicks = WAIT_TICKS;
        runTicks = 0;
        lastDistance = Double.NaN;
        pendingReset = true;
        if (!initial) unpress();
    }

    private void teleportToStart() {
        if (mc.player == null || teleportCommand == null) return;
        ChatUtils.sendPlayerMsg(teleportCommand, false);
    }

    @EventHandler
    private void onTick(TickEvent.Pre event) {
        if (python == null || mc.player == null) return;

        refreshTargets();
        if (startVec == null || goalVec == null) return;
        if (cachedStart.equals(cachedGoal)) {
            error("Start and goal points are identical. Stopping module.");
            toggle();
            return;
        }

        if (phase == EpisodePhase.WAITING) {
            unpress();
            if (waitTicks > 0) waitTicks--;
            if (waitTicks <= 0) {
                phase = EpisodePhase.RUNNING;
                runTicks = 0;
                lastDistance = mc.player.getPos().distanceTo(goalVec);
            }
            return;
        }

        Vec3d pos = mc.player.getPos();
        Vec3d vel = mc.player.getVelocity();

        runTicks++;
        double distance = pos.distanceTo(goalVec);
        double progress = Double.isNaN(lastDistance) ? 0.0 : lastDistance - distance;
        lastDistance = distance;

        boolean reached = distance <= goalRadius.get();
        boolean timedOut = runTicks >= MAX_RUN_TICKS;
        boolean done = reached || timedOut;

        JsonObject obj = new JsonObject();
        obj.addProperty("px", pos.x);
        obj.addProperty("py", pos.y);
        obj.addProperty("pz", pos.z);
        obj.addProperty("vx", vel.x);
        obj.addProperty("vy", vel.y);
        obj.addProperty("vz", vel.z);
        obj.addProperty("pitch", mc.player.getPitch());
        obj.addProperty("yaw", mc.player.getYaw());
        obj.addProperty("sprinting", mc.player.isSprinting());
        obj.addProperty("gx", goalVec.x);
        obj.addProperty("gy", goalVec.y);
        obj.addProperty("gz", goalVec.z);
        obj.addProperty("distance", distance);
        obj.addProperty("progress", progress);
        obj.addProperty("time_elapsed", runTicks / 20.0);
        obj.addProperty("time_remaining", Math.max(0, MAX_RUN_TICKS - runTicks) / 20.0);
        obj.addProperty("done", done);
        obj.addProperty("reached", reached);
        obj.addProperty("timeout", timedOut);
        obj.addProperty("episode", episodeCounter);
        if (pendingReset) {
            obj.addProperty("reset", true);
            pendingReset = false;
        }

        try {
            writer.write(obj.toString());
            writer.newLine();
            writer.flush();

            String line = reader.readLine();
            if (line == null) throw new IOException("Python process terminated.");
            line = line.trim();
            if (!line.isEmpty()) {
                try {
                    JsonObject act = JsonParser.parseString(line).getAsJsonObject();
                    int move = act.has("move") ? act.get("move").getAsInt() : 0;
                    int strafe = act.has("strafe") ? act.get("strafe").getAsInt() : 0;
                    int yawA = act.has("yaw") ? act.get("yaw").getAsInt() : 0;
                    int pitchA = act.has("pitch") ? act.get("pitch").getAsInt() : 0;
                    int jump = act.has("jump") ? act.get("jump").getAsInt() : 0;
                    applyCompositeAction(move, strafe, yawA, pitchA, jump);
                } catch (Exception ignored) {
                    applyCompositeAction(0, 0, 0, 0, 0);
                }
            }

            if (done) {
                handleEpisodeEnd(reached, timedOut, distance);
            }
        } catch (Exception e) {
            error("Python error: %s", e.getMessage());
            toggle();
        }
    }

    private void handleEpisodeEnd(boolean reached, boolean timedOut, double finalDistance) {
        if (reached) {
            successes++;
            info("Goal reached in %.2fs (episode %d).", runTicks / 20.0, episodeCounter + 1);
        } else if (timedOut) {
            info("Episode %d timed out with %.2f blocks remaining.", episodeCounter + 1, finalDistance);
        } else {
            info("Episode %d ended.", episodeCounter + 1);
        }

        episodeCounter++;
        beginWaitingPhase(false);
        teleportToStart();
    }

    @EventHandler
    private void onRender(Render3DEvent event) {
        if (cachedStart == null || cachedGoal == null) return;

        Box startBox = new Box(cachedStart.getX(), cachedStart.getY(), cachedStart.getZ(), cachedStart.getX() + 1, cachedStart.getY() + 1, cachedStart.getZ() + 1);
        Box goalBox = new Box(cachedGoal.getX(), cachedGoal.getY(), cachedGoal.getZ(), cachedGoal.getX() + 1, cachedGoal.getY() + 1, cachedGoal.getZ() + 1);

        event.renderer.box(startBox, Color.ORANGE, Color.ORANGE, ShapeMode.Lines, 0);
        event.renderer.box(goalBox, Color.GREEN, Color.GREEN, ShapeMode.Lines, 0);
    }

    @Override
    public WWidget getWidget(GuiTheme theme) {
        WVerticalList list = theme.verticalList();

        WButton startButton = list.add(theme.button("Set start point"))
            .expandX()
            .widget();
        startButton.action = this::setStartFromPlayer;

        WButton goalButton = list.add(theme.button("Set goal point"))
            .expandX()
            .widget();
        goalButton.action = this::setGoalFromPlayer;

        return list;
    }

    private void setStartFromPlayer() {
        if (mc.player == null) return;
        BlockPos pos = BlockPos.ofFloored(mc.player.getX(), mc.player.getY() - 0.01, mc.player.getZ());
        startPoint.set(pos);
        info("Start point set to %d %d %d", pos.getX(), pos.getY(), pos.getZ());
        refreshTargets();
    }

    private void setGoalFromPlayer() {
        if (mc.player == null) return;
        BlockPos pos = BlockPos.ofFloored(mc.player.getX(), mc.player.getY() - 0.01, mc.player.getZ());
        goalPoint.set(pos);
        info("Goal point set to %d %d %d", pos.getX(), pos.getY(), pos.getZ());
        refreshTargets();
    }

    @Override
    public String getInfoString() {
        if (phase == EpisodePhase.RUNNING) {
            return String.format("%.1fs", runTicks / 20.0);
        }
        return "wait";
    }

    private enum EpisodePhase {
        WAITING,
        RUNNING
    }
}
