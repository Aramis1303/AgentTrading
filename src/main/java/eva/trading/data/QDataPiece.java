/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package eva.trading.data;

/**
 *
 * @author Eva
 */
public class QDataPiece {
    
    // Дата свечи
    private long time;
    // Значения свечи
    private double open;
    private double close;
    private double high;
    private double low;
    private double volume;
    // Значения индикаторов
    private double rsi_3;
    private double rsi_7;
    private double rsi_13;
    private double rsi_21;
    private double rsi_31;
    
    private double stohasticMain_3;
    private double stohasticSignal_3;
    private double stohasticMain_7;
    private double stohasticSignal_7;
    private double stohasticMain_13;
    private double stohasticSignal_13;
    private double stohasticMain_21;
    private double stohasticSignal_21;
    private double stohasticMain_31;
    private double stohasticSignal_31;
    
    private double adxMain_3;
    private double adxPlus_3;
    private double adxMinus_3;
    private double adxMain_7;
    private double adxPlus_7;
    private double adxMinus_7;
    private double adxMain_13;
    private double adxPlus_13;
    private double adxMinus_13;
    private double adxMain_21;
    private double adxPlus_21;
    private double adxMinus_21;
    private double adxMain_31;
    private double adxPlus_31;
    private double adxMinus_31;
    
    private double ema_3;
    private double ema_7;
    private double ema_13;
    private double ema_21;
    private double ema_31;
    
    public QDataPiece (long time, double open, double close, double high, double low, double volume,
                       double rsi_3, double rsi_7, double rsi_13, double rsi_21, double rsi_31, 
                       double stohasticMain_3, double stohasticMain_7, double stohasticMain_13, double stohasticMain_21, double stohasticMain_31, 
                       double stohasticSignal_3, double stohasticSignal_7, double stohasticSignal_13, double stohasticSignal_21, double stohasticSignal_31, 
                       double adxMain_3, double adxMain_7, double adxMain_13, double adxMain_21, double adxMain_31, 
                       double adxPlus_3, double adxPlus_7, double adxPlus_13, double adxPlus_21, double adxPlus_31, 
                       double adxMinus_3, double adxMinus_7, double adxMinus_13, double adxMinus_21, double adxMinus_31, 
                       double ema_3, double ema_7, double ema_13, double ema_21, double ema_31) 
    {
        // Дата свечи
        this.time = time;
        // Значения свечи
        this.open = open;
        this.close = close;
        this.high = high;
        this.low = low;
        this.volume = volume;
        // Значения индикаторов
        this.rsi_3 = rsi_3;
        this.rsi_7 = rsi_7;
        this.rsi_13 = rsi_13;
        this.rsi_21 = rsi_21;
        this.rsi_31 = rsi_31;

        this.stohasticMain_3 = stohasticMain_3;
        this.stohasticSignal_3 = stohasticSignal_3;
        this.stohasticMain_7 = stohasticMain_7;
        this.stohasticSignal_7 = stohasticSignal_7;
        this.stohasticMain_13 = stohasticMain_13;
        this.stohasticSignal_13 = stohasticSignal_13;
        this.stohasticMain_21 = stohasticMain_21;
        this.stohasticSignal_21 = stohasticSignal_21;
        this.stohasticMain_31 = stohasticMain_31;
        this.stohasticSignal_31 = stohasticSignal_31;

        this.adxMain_3 = adxMain_3;
        this.adxPlus_3 = adxPlus_3;
        this.adxMinus_3 = adxMinus_3;
        this.adxMain_7 = adxMain_7;
        this.adxPlus_7 = adxPlus_7;
        this.adxMinus_7 = adxMinus_7;
        this.adxMain_13 = adxMain_13;
        this.adxPlus_13 = adxPlus_13;
        this.adxMinus_13 = adxMinus_13;
        this.adxMain_21 = adxMain_21;
        this.adxPlus_21 = adxPlus_21;
        this.adxMinus_21 = adxMinus_21;
        this.adxMain_31 = adxMain_31;
        this.adxPlus_31 = adxPlus_31;
        this.adxMinus_31 = adxMinus_31;

        this.ema_3 = ema_3;
        this.ema_7 = ema_7;
        this.ema_13 = ema_13;
        this.ema_21 = ema_21;
        this.ema_31 = ema_31;
    }  
    
    @Override
    public boolean equals (Object o) {
        return ((QDataPiece)o).time == this.time;
    }

    public long getTime() {
        return time;
    }

    public double getOpen() {
        return open;
    }

    public double getClose() {
        return close;
    }

    public double getHigh() {
        return high;
    }

    public double getLow() {
        return low;
    }

    public double getVolume() {
        return volume;
    }

    public double getRsi_3() {
        return rsi_3;
    }

    public double getRsi_7() {
        return rsi_7;
    }

    public double getRsi_13() {
        return rsi_13;
    }

    public double getRsi_21() {
        return rsi_21;
    }

    public double getRsi_31() {
        return rsi_31;
    }

    public double getStohasticMain_3() {
        return stohasticMain_3;
    }

    public double getStohasticSignal_3() {
        return stohasticSignal_3;
    }

    public double getStohasticMain_7() {
        return stohasticMain_7;
    }

    public double getStohasticSignal_7() {
        return stohasticSignal_7;
    }

    public double getStohasticMain_13() {
        return stohasticMain_13;
    }

    public double getStohasticSignal_13() {
        return stohasticSignal_13;
    }

    public double getStohasticMain_21() {
        return stohasticMain_21;
    }

    public double getStohasticSignal_21() {
        return stohasticSignal_21;
    }

    public double getStohasticMain_31() {
        return stohasticMain_31;
    }

    public double getStohasticSignal_31() {
        return stohasticSignal_31;
    }

    public double getAdxMain_3() {
        return adxMain_3;
    }

    public double getAdxPlus_3() {
        return adxPlus_3;
    }

    public double getAdxMinus_3() {
        return adxMinus_3;
    }

    public double getAdxMain_7() {
        return adxMain_7;
    }

    public double getAdxPlus_7() {
        return adxPlus_7;
    }

    public double getAdxMinus_7() {
        return adxMinus_7;
    }

    public double getAdxMain_13() {
        return adxMain_13;
    }

    public double getAdxPlus_13() {
        return adxPlus_13;
    }

    public double getAdxMinus_13() {
        return adxMinus_13;
    }

    public double getAdxMain_21() {
        return adxMain_21;
    }

    public double getAdxPlus_21() {
        return adxPlus_21;
    }

    public double getAdxMinus_21() {
        return adxMinus_21;
    }

    public double getAdxMain_31() {
        return adxMain_31;
    }

    public double getAdxPlus_31() {
        return adxPlus_31;
    }

    public double getAdxMinus_31() {
        return adxMinus_31;
    }

    public double getEma_3() {
        return ema_3;
    }

    public double getEma_7() {
        return ema_7;
    }

    public double getEma_13() {
        return ema_13;
    }

    public double getEma_21() {
        return ema_21;
    }

    public double getEma_31() {
        return ema_31;
    }
    
    
}
