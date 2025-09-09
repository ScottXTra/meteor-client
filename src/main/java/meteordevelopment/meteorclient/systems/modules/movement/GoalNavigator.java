package your.addon.modules.movement;

import meteordevelopment.meteorclient.events.render.Render3DEvent;
import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.renderer.ShapeMode;
import meteordevelopment.meteorclient.settings.*;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.utils.render.color.SettingColor;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.util.math.MathHelper;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Locale;

public class GoalNavigator extends Module {
    private Process python;
    private BufferedWriter writer;
    private BufferedReader reader;

    private final SettingGroup sgGeneral = settings.createGroup("General");
    private final SettingGroup sgRender  = settings.createGroup("Render");

    private final Setting<Double> goalDistance = sgGeneral.add(new DoubleSetting.Builder()
        .name("goal-distance")
        .description("Distance in blocks in front of you when (re)setting goal.")
        .defaultValue(5.0)
        .min(1.0).sliderRange(1.0, 20.0)
        .build()
    );

    private final Setting<Integer> episodeTicks = sgGeneral.add(new IntSetting.Builder()
        .name("episode-length-ticks")
        .description("Max ticks before an episode ends.")
        .defaultValue(40)
        .min(10).sliderRange(10, 400)
        .build()
    );

    private final Setting<Double> successRadius = sgGeneral.add(new DoubleSetting.Builder()
        .name("success-radius")
        .description("Episode ends early when within this distance of goal.")
        .defaultValue(0.8)
        .min(0.2).sliderRange(0.2, 3.0)
        .build()
    );

    private final Setting<Boolean> newGoalOnSuccess = sgGeneral.add(new BoolSetting.Builder()
        .name("new-goal-on-success")
        .description("Pick a new goal ahead when you reach the current one.")
        .defaultValue(true)
        .build()
    );

    private final Setting<Boolean> sprint = sgGeneral.add(new BoolSetting.Builder()
        .name("auto-sprint")
        .description("Hold sprint key while moving forward.")
        .defaultValue(true)
        .build()
    );

    private final Setting<Boolean> allowJump = sgGeneral.add(new BoolSetting.Builder()
        .name("allow-jump")
        .description("Let the agent press jump (uses forward+jump to climb 1-block steps).")
        .defaultValue(true)
        .build()
    );

    private final Setting<Double> turnStepDeg = sgGeneral.add(new DoubleSetting.Builder()
        .name("turn-step-deg")
        .description("How many degrees to rotate on a single turn action.")
        .defaultValue(7.5)
        .min(1.0).sliderRange(1.0, 30.0)
        .build()
    );

    private final Setting<String> pythonPath = sgGeneral.add(new StringSetting.Builder()
        .name("python-path")
        .description("Interpreter to run goal_navigator.py (use python on Windows, python3 on Linux/macOS).")
        .defaultValue(isWindows() ? "python" : "python3")
        .build()
    );

    private final Setting<Boolean> render = sgRender.add(new BoolSetting.Builder()
        .name("render")
        .description("Renders a box at the goal position.")
        .defaultValue(true)
        .build()
    );

    private final Setting<ShapeMode> shapeMode = sgRender.add(new EnumSetting.Builder<ShapeMode>()
        .name("shape-mode")
        .description("How the goal box is rendered.")
        .defaultValue(ShapeMode.Both)
        .visible(render::get)
        .build()
    );

    private final Setting<SettingColor> sideColor = sgRender.add(new ColorSetting.Builder()
        .name("side-color")
        .description("The side color of the goal box.")
        .defaultValue(new SettingColor(255, 0, 0, 75))
        .visible(() -> render.get() && shapeMode.get().sides())
        .build()
    );

    private final Setting<SettingColor> lineColor = sgRender.add(new ColorSetting.Builder()
        .name("line-color")
        .description("The line color of the goal box.")
        .defaultValue(new SettingColor(255, 0, 0, 255))
        .visible(() -> render.get() && shapeMode.get().lines())
        .build()
    );

    // goal (center of 1x1x1 box)
    private double goalX, goalY, goalZ;
    private int ticks;
    private String lastAction = "none"; // reuse if Python isn't ready

    // Actions understood by the Python script
    // forward, back, left, right, turn_left, turn_right, jump, none
    private static final String[] ACTIONS = {
        "forward","back","left","right","turn_left","turn_right","jump","none"
    };

    public GoalNavigator() {
        super(Categories.Movement, "goal-navigator", "Navigates to a goal using a Python RL model.");
    }

    @Override
    public void onActivate() {
        ClientPlayerEntity p = mc.player;
        if (p == null) return;

        setGoalAhead();
        ticks = 0;
        lastAction = "none";

        try {
            InputStream script = GoalNavigator.class.getResourceAsStream("/scripts/goal_navigator.py");
            if (script == null) {
                error("Script not found in resources: /scripts/goal_navigator.py");
                toggle();
                return;
            }

            File tmp = File.createTempFile("goal_navigator", ".py");
            tmp.deleteOnExit();
            Files.copy(script, tmp.toPath(), StandardCopyOption.REPLACE_EXISTING);

            ProcessBuilder pb = new ProcessBuilder(pythonPath.get(), "-u", tmp.getAbsolutePath());
            pb.redirectErrorStream(true);
            python = pb.start();
            writer = new BufferedWriter(new OutputStreamWriter(python.getOutputStream()));
            reader = new BufferedReader(new InputStreamReader(python.getInputStream()));
            info("Launched Python: " + pythonPath.get());
        } catch (IOException e) {
            error("Failed to start python: {}", e.getMessage());
            toggle();
        }
    }

    @Override
    public void onDeactivate() {
        clearMovementKeys();
        if (python != null) {
            python.destroy();
            python = null;
        }
        writer = null;
        reader = null;
    }

    @EventHandler
    private void onTick(TickEvent.Post event) {
        if (mc.player == null) return;
        if (mc.currentScreen != null) { // don't press keys in menus
            clearMovementKeys();
            return;
        }
        if (python == null || writer == null || reader == null) return;
        if (!python.isAlive()) {
            error("Python process exited.");
            toggle();
            return;
        }

        ClientPlayerEntity p = mc.player;

        // Episode termination: close enough OR time exceeded
        double dx = goalX - p.getX();
        double dz = goalZ - p.getZ();
        double distance = Math.hypot(dx, dz);

        boolean success = distance <= successRadius.get();
        boolean timeUp  = ++ticks >= episodeTicks.get();
        boolean done = success || timeUp;
        if (done) ticks = 0;

        // yaw delta to face goal (Meteor/Yarn yaw -> forward is (-sin, cos))
        float yaw = p.getYaw();
        float targetYaw = (float) Math.toDegrees(Math.atan2(-dx, dz));
        float deltaYaw = MathHelper.wrapDegrees(targetYaw - yaw); // [-180, 180]

        try {
            // x,z,vx,vz,gx,gz, deltaYaw(deg), horizCollision, done
            String line = String.format(Locale.ROOT, "%f,%f,%f,%f,%f,%f,%f,%d,%d\n",
                p.getX(), p.getZ(),
                p.getVelocity().x, p.getVelocity().z,
                goalX, goalZ,
                (double) deltaYaw,
                p.horizontalCollision ? 1 : 0,
                done ? 1 : 0
            );
            writer.write(line);
            writer.flush();

            // read action (non-blocking)
            if (reader.ready()) {
                String resp = reader.readLine();
                if (resp != null && !resp.isEmpty()) {
                    String[] parts = resp.split("\\|", 2);
                    String act = parts[0].trim();
                    if (!act.isEmpty()) lastAction = act;
                    if (parts.length > 1) {
                        String msg = parts[1].trim();
                        if (!msg.isEmpty()) info(msg);
                    }
                }
            }

            applyAction(lastAction);

            if (done && success && newGoalOnSuccess.get()) {
                setGoalAhead(); // pick a fresh target ahead
            }
        } catch (IOException e) {
            error("Python communication error: {}", e.getMessage());
            toggle();
        }
    }

    @EventHandler
    private void onRender(Render3DEvent event) {
        if (!render.get()) return;
        event.renderer.box(
            goalX - 0.5, goalY, goalZ - 0.5,
            goalX + 0.5, goalY + 1.0, goalZ + 0.5,
            sideColor.get(), lineColor.get(), shapeMode.get(), 0
        );
    }

    private void setGoalAhead() {
        ClientPlayerEntity p = mc.player;
        if (p == null) return;
        double yawRad = Math.toRadians(p.getYaw(1.0f));
        double dist = goalDistance.get();
        goalX = p.getX() + (-Math.sin(yawRad)) * dist;
        goalZ = p.getZ() + ( Math.cos(yawRad)) * dist;
        goalY = Math.floor(p.getY()); // simple same-height target
    }

    private void clearMovementKeys() {
        mc.options.forwardKey.setPressed(false);
        mc.options.backKey.setPressed(false);
        mc.options.leftKey.setPressed(false);
        mc.options.rightKey.setPressed(false);
        try { mc.options.jumpKey.setPressed(false); } catch (Throwable ignored) {}
        try { mc.options.sprintKey.setPressed(false); } catch (Throwable ignored) {}
    }

    private void applyAction(String action) {
        // Always clear first to avoid sticky keys
        clearMovementKeys();

        switch (action) {
            case "forward":
                mc.options.forwardKey.setPressed(true);
                if (sprint.get()) try { mc.options.sprintKey.setPressed(true); } catch (Throwable ignored) {}
                break;
            case "back":
                mc.options.backKey.setPressed(true);
                break;
            case "left":
                mc.options.leftKey.setPressed(true);
                break;
            case "right":
                mc.options.rightKey.setPressed(true);
                break;
            case "turn_left":
                if (mc.player != null) mc.player.setYaw(mc.player.getYaw() - turnStepDeg.get().floatValue());
                break;
            case "turn_right":
                if (mc.player != null) mc.player.setYaw(mc.player.getYaw() + turnStepDeg.get().floatValue());
                break;
            case "jump":
                if (!allowJump.get()) break;
                // helpful default: jump while moving forward to step up blocks
                mc.options.forwardKey.setPressed(true);
                try { mc.options.jumpKey.setPressed(true); } catch (Throwable ignored) {}
                if (sprint.get()) try { mc.options.sprintKey.setPressed(true); } catch (Throwable ignored) {}
                break;
            case "none":
            default:
                // no-op
                break;
        }
    }

    private static boolean isWindows() {
        String os = System.getProperty("os.name", "generic").toLowerCase(Locale.ROOT);
        return os.contains("win");
    }
}
