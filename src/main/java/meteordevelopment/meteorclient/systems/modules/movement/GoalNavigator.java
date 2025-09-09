package meteordevelopment.meteorclient.systems.modules.movement;

import meteordevelopment.meteorclient.events.render.Render3DEvent;
import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.renderer.ShapeMode;
import meteordevelopment.meteorclient.settings.*;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.utils.render.color.SettingColor;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.client.network.ClientPlayerEntity;

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
        .description("Distance in blocks from current facing direction.")
        .defaultValue(5.0)
        .min(1.0).sliderRange(1.0, 15.0)
        .build()
    );

    private final Setting<Integer> episodeTicks = sgGeneral.add(new IntSetting.Builder()
        .name("episode-length-ticks")
        .description("Max ticks before an episode ends.")
        .defaultValue(40) // 2 seconds @ 20tps
        .min(10).sliderRange(10, 200)
        .build()
    );

    private final Setting<Double> successRadius = sgGeneral.add(new DoubleSetting.Builder()
        .name("success-radius")
        .description("Episode ends early when within this distance of goal.")
        .defaultValue(0.8)
        .min(0.2).sliderRange(0.2, 3.0)
        .build()
    );

    private final Setting<Boolean> sprint = sgGeneral.add(new BoolSetting.Builder()
        .name("auto-sprint")
        .description("Hold sprint key while moving forward.")
        .defaultValue(true)
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

    private double goalX, goalY, goalZ;
    private int ticks;
    private String lastAction = "none"; // reuse if Python isn't ready

    public GoalNavigator() {
        super(Categories.Movement, "goal-navigator", "Navigates to a goal using a Python RL model.");
    }

    @Override
    public void onActivate() {
        ClientPlayerEntity player = mc.player;
        if (player == null) return;

        // Goal N blocks ahead in facing direction
        double yawRad = Math.toRadians(player.getYaw(1.0f));
        double dist = goalDistance.get();
        goalX = player.getX() + (-Math.sin(yawRad)) * dist;
        goalZ = player.getZ() + ( Math.cos(yawRad)) * dist;
        goalY = player.getY();
        ticks = 0;
        lastAction = "none";

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
        // clear keys
        if (mc.player != null) {
            clearMovementKeys();
        }
        // tear down python
        if (python != null) {
            python.destroy();
            python = null;
        }
        writer = null;
        reader = null;
    }

    @EventHandler
    private void onTick(TickEvent.Post event) {
        if (python == null || writer == null || reader == null || mc.player == null) return;
        if (!python.isAlive()) {
            error("Python process exited.");
            toggle();
            return;
        }

        double x  = mc.player.getX();
        double z  = mc.player.getZ();
        double vx = mc.player.getVelocity().x;
        double vz = mc.player.getVelocity().z;

        // Episode termination: close enough OR time exceeded
        double dx = goalX - x;
        double dz = goalZ - z;
        double distance = Math.hypot(dx, dz);

        boolean done = distance <= successRadius.get() || ++ticks >= episodeTicks.get();
        if (done) ticks = 0;

        try {
            // send observation
            String line = String.format(Locale.ROOT, "%f,%f,%f,%f,%f,%f,%d\n",
                    x, z, vx, vz, goalX, goalZ, done ? 1 : 0);
            writer.write(line);
            writer.flush();

            // read action and optional chat message if available (non-blocking)
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
            goalX + 0.5, goalY + 1, goalZ + 0.5,
            sideColor.get(), lineColor.get(), shapeMode.get(), 0
        );
    }

    private void clearMovementKeys() {
        mc.options.forwardKey.setPressed(false);
        mc.options.backKey.setPressed(false);
        mc.options.leftKey.setPressed(false);
        mc.options.rightKey.setPressed(false);
        try {
            mc.options.sprintKey.setPressed(false);
        } catch (Throwable ignored) {
            // some versions/keymaps may not expose sprintKey
        }
    }

    private void applyAction(String action) {
        // Always clear first to avoid sticky keys
        clearMovementKeys();

        switch (action) {
            case "forward":
                mc.options.forwardKey.setPressed(true);
                if (sprint.get()) {
                    try { mc.options.sprintKey.setPressed(true); } catch (Throwable ignored) {}
                }
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
            case "none":
            default:
                // no-op
                break;
        }
    }
}
