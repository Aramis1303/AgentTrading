/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package eva.trading;

import eva.trading.agent.AgentR;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


import eva.trading.agent.AgentRQuorum;
/**
 *
 * @author username
 */

public class Starter {
    
    static List <AgentR> agents = new ArrayList<>();
    
    public static void main(String args[]) throws IOException, InterruptedException {
        
        agents.add(new AgentR("sber_5"));
        
    }
    
    public static void stopAll() {
        for (AgentR aq: agents) {
            aq.stop();
        }
    }
    
}
