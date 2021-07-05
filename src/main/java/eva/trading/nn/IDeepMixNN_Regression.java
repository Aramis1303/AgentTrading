/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package eva.trading.nn;

import eva.trading.data.QDataset;
import eva.trading.data.QState;
import java.io.IOException;
import org.deeplearning4j.ui.api.UIServer;

/**
 *
 * @author username
 */
public interface IDeepMixNN_Regression {
    
    public void putData (QDataset qds);
    public void setWebUI (UIServer uiServer);
    public void prepareDatasets();
    public void trainingNetwork (int epoch);
    public void testNetworkRegression ();
    
    public double score();
    public double accuracy();
    public double precision();
    public double recall();
    public double mae();
    public double mqe();
    
    public double prediction (QState qs);
    public void saveCurrentNeuralNetwork (String pathBinFile) throws IOException;
    public void saveBestNeuralNetwork (String pathBinFile) throws IOException;
    public void dumpNetwork();
    public boolean dumpIsNull();
    
}
