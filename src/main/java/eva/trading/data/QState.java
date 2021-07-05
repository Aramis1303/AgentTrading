/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package eva.trading.data;

import eva.trading.Settings;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author username
 */
// Сюда записываем последовательность рынка, отмечаем находимся ли мы в сделке, действие которое было совершено и конечную оценку данного действия
public class QState {
    
    private List <QDataPiece> series = null;    // История до текущего состояния
    
    private double rewBuy = 0;                  // Оценка для действия Buy
    private double rewSell = 0;                 // Оценка для действия Sell
    private double rewClose = 0;                // Оценка для действия Close
    private double rewWait = 0;                 // Оценка для действия Wait
    
    
    public QState() {
        series = new ArrayList<>();
    }
    
    public List<QDataPiece> getSeries() {
        return series;
    }
    public void addSeries(QDataPiece qds) {
        series.add(qds);
    }

    public double getRewBuy() {
        return rewBuy;
    }
    public void setRewBuy(double e) {
        rewBuy = e;
        if (rewBuy > 1) rewBuy = 1;
        if (rewBuy < -1) rewBuy = -1;
    }
    
    public double getRewSell() {
        return rewSell;
    }
    public void setRewSell(double e) {
        rewSell = e;
        if (rewSell > 1) rewSell = 1;
        if (rewSell < -1) rewSell = -1;
    }
    
    public double getRewWait() {
        return rewWait;
    }
    public void setRewWait(double e) {
        rewWait = e;
        if (rewWait > 1) rewWait = 1;
        if (rewWait < -1) rewWait = -1;
    }
    
    public double getRewClose() {
        return rewClose;
    }
    public void setRewClose(double e) {
        rewClose = e;
        if (rewClose > 1) rewClose = 1;
        if (rewClose < -1) rewClose = -1;
    }
    
    
    
    
    
    
}
