/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package eva.trading.data;

import eva.trading.data.EDealStates;
import eva.trading.Settings;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author username
 */
public class QDataset implements Comparable <QDataset> {
    
    private long startDealDT = 0;   // Время входа в сделку
    private long stopDealDT = 0;    // Время выхода из сделки
    
    private List <QState> qStates = null;       // Набор состояний
    
    private double reward = 0;
    
    private EDealStates eds;
    
    public QDataset(EDealStates eds) {
        qStates = new ArrayList<>();
        this.eds = eds;
    }
    
    public void addReward(double r) {
        reward += r;
    }
    public double getReward() {
        return reward;
    }
    
    
    public void addQState(QState qs) {
        qStates.add(qs);
    }
    public List<QState> getQStates() {
        return qStates;
    }
    
    // Расчет каждого шага сделки
    public void calculatEval() {
        /*double r = reward / qStates.size();
        switch (eds) {
            case LONG: {
                for (int i = 0; i < qStates.size(); i++) {
                    if (reward > 0) {
                        // Оченка высокаю первые шаги, дальше быстро падает к 0
                        qStates.get(i).setRewBuy ((qStates.size() -i) * r);
                        qStates.get(i).setRewSell((-1) * (qStates.size() -i) * r * (Settings.PERCENT_STOP_LOSS));
                        qStates.get(i).setRewWait(Math.tanh(qStates.size() - (i + 2)) / 2 -0.45);
                        qStates.get(i).setRewClose(Math.tanh((i + 2) - qStates.size()) / 2 +0.45);
                        
                        //qStates.get(i).setRewSell(Math.tanh((-1) * reward / (i+1)));
                        //qStates.get(i).setRewWait(Math.tanh(qStates.size() - (i + 2)) / 2 -0.45);
                        //qStates.get(i).setRewClose(Math.tanh((i + 2) - qStates.size()) / 2 +0.45);
                        //qStates.get(i).setRewBuy (Math.tanh(2 * reward / (i+1)));
                    }
                    else {
                        // Если это штраф по стопу, то buy в
                        //qStates.get(i).setRewBuy(Math.tanh((2) * reward / (i+1)));
                        
                        qStates.get(i).setRewBuy((qStates.size() -i) * r);
                        qStates.get(i).setRewSell(0.0);
                        qStates.get(i).setRewWait(0.0);
                        qStates.get(i).setRewClose(0.0);
                    }
                }
            }; break;
            case SHORT: {
                for (int i = 0; i < qStates.size(); i++) {
                    if (reward > 0) {
                        qStates.get(i).setRewSell((qStates.size() -i) * r);
                        qStates.get(i).setRewBuy ((-1) * (qStates.size() -i) * r * (Settings.PERCENT_STOP_LOSS));
                        qStates.get(i).setRewWait(Math.tanh(qStates.size() - (i + 2)) / 2 -0.45);
                        qStates.get(i).setRewClose(Math.tanh((i + 2) - qStates.size()) / 2 +0.45);
                        
                        //qStates.get(i).setRewSell(Math.tanh(2 * reward / (i+1)));
                        //qStates.get(i).setRewBuy (Math.tanh((-1) * reward / (i+1)));
                        //qStates.get(i).setRewWait(Math.tanh(qStates.size() - (i + 2)) / 2 -0.45);
                        //qStates.get(i).setRewClose(Math.tanh((i + 2) - qStates.size()) / 2 +0.45);
                    }
                    else {
                        // Если это штраф по стопу, то buy в
                        //qStates.get(i).setRewSell(Math.tanh((2) * reward / (i+1)));
                        
                        qStates.get(i).setRewSell((qStates.size() -i) * r);
                        qStates.get(i).setRewBuy(0.0);
                        qStates.get(i).setRewWait(0.0);
                        qStates.get(i).setRewClose(0.0);
                    }
                }
            }; break;
            default: {
                for (int i = 0; i < qStates.size(); i++) {
                    qStates.get(i).setRewSell(0.0);
                    qStates.get(i).setRewBuy(0.0);
                    qStates.get(i).setRewWait(0.0);
                    qStates.get(i).setRewClose(0.0);
                }
            }
            
        }
        */
    }
    
    public EDealStates getState() {
        return eds;
    }

    @Override
    public int compareTo(QDataset qds) {
        if (this.getReward() > qds.getReward()) {
            return 1;
        }
        else return -1;
    }
}
