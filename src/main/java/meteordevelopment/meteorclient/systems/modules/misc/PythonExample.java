/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.misc;

import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.systems.modules.Categories;
import org.python.util.PythonInterpreter;
import org.python.core.PyFunction;
import org.python.core.PyObject;
import org.python.core.Py;

import java.io.InputStream;

public class PythonExample extends Module {
    public PythonExample() {
        super(Categories.Misc, "python-example", "Demonstrates calling a Python function from Java.");
    }

    @Override
    public void onActivate() {
        try (PythonInterpreter py = new PythonInterpreter();
             InputStream script = PythonExample.class.getResourceAsStream("/scripts/add_numbers.py")) {

            if (script == null) {
                error("Could not find Python script.");
            } else {
                py.execfile(script);
                PyFunction add = (PyFunction) py.get("add", PyFunction.class);
                PyObject result = add.__call__(Py.newInteger(2), Py.newInteger(3));
                info("2 + 3 = " + result.toString() );
            }
        } catch (Exception e) {
            error("Python error: {}", e.getMessage());
        }

        toggle();
    }
}
