package meteordevelopment.meteorclient.systems.modules.movement;

import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.settings.DoubleSetting;
import meteordevelopment.meteorclient.settings.Setting;
import meteordevelopment.meteorclient.settings.SettingGroup;
import net.minecraft.util.math.Vec3d;

import java.io.*;

public class PythonQLearning extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private final Setting<Double> goalX = sgGeneral.add(new DoubleSetting.Builder()
        .name("goal-x")
        .description("Goal X coordinate")
        .defaultValue(0)
        .build());

    private final Setting<Double> goalY = sgGeneral.add(new DoubleSetting.Builder()
        .name("goal-y")
        .description("Goal Y coordinate")
        .defaultValue(0)
        .build());

    private final Setting<Double> goalZ = sgGeneral.add(new DoubleSetting.Builder()
        .name("goal-z")
        .description("Goal Z coordinate")
        .defaultValue(0)
        .build());

    private Process process;
    private BufferedWriter writer;
    private BufferedReader reader;
    private Thread thread;

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
        } catch (IOException e) {
            error("Failed to start Python: {}", e.getMessage());
            toggle();
            return;
        }

        thread = new Thread(this::loop, "python-qlearning-loop");
        thread.start();
    }

    private void loop() {
        try {
            while (isActive() && mc.player != null && !Thread.interrupted()) {
                Vec3d p = mc.player.getPos();
                String msg = String.format("{\"player\":{\"x\":%f,\"y\":%f,\"z\":%f},\"goal\":{\"x\":%f,\"y\":%f,\"z\":%f}}\n",
                    p.x, p.y, p.z, goalX.get(), goalY.get(), goalZ.get());
                writer.write(msg);
                writer.flush();

                String action = reader.readLine();
                if (action == null) break;
                mc.execute(() -> applyAction(action.trim()));
                Thread.sleep(50);
            }
        } catch (Exception e) {
            error("Python communication error: {}", e.getMessage());
        }
        mc.execute(this::toggle);
    }

    private void applyAction(String action) {
        mc.options.forwardKey.setPressed(action.equals("forward"));
        mc.options.backKey.setPressed(action.equals("back"));
        mc.options.leftKey.setPressed(action.equals("left"));
        mc.options.rightKey.setPressed(action.equals("right"));
    }

    @Override
    public void onDeactivate() {
        mc.options.forwardKey.setPressed(false);
        mc.options.backKey.setPressed(false);
        mc.options.leftKey.setPressed(false);
        mc.options.rightKey.setPressed(false);

        if (thread != null) thread.interrupt();
        if (process != null) process.destroy();
    }
}
