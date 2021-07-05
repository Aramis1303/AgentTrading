/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package eva.trading.agent;

import java.io.IOException;

/**
 *
 * @author Admin
 */
public interface IAgent extends Runnable {
    public void compileDatasets();
    public void toLearn() throws IOException;
    public void stop();
}
