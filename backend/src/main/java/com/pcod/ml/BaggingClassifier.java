package com.pcod.ml;

import smile.classification.DecisionTree;
import smile.data.Tuple;

import java.io.Serializable;

/**
 * Bagging (Bootstrap Aggregating) classifier.
 *
 * <p>Smile 6.3.0 ships no Bagging classifier class, so bagging is implemented
 * here directly from its textbook definition, on top of Smile's
 * {@link DecisionTree}: {@code nTrees} decision trees are each trained on a
 * bootstrap sample (n rows drawn with replacement from the training set) and
 * the final prediction is the majority vote. The vote fraction for class 1 is
 * returned as the probability of PCOS. This is a genuine ensemble model, not
 * a wrapper around Random Forest (which additionally randomises the features
 * tried at each split).</p>
 */
public class BaggingClassifier implements Serializable {

    private static final long serialVersionUID = 1L;

    private final DecisionTree[] trees;

    public BaggingClassifier(DecisionTree[] trees) {
        if (trees == null || trees.length == 0) {
            throw new IllegalArgumentException("Bagging needs at least one tree");
        }
        this.trees = trees.clone();
    }

    public int numTrees() {
        return trees.length;
    }

    /** Majority-vote prediction. */
    public int predict(Tuple x) {
        double[] posteriori = new double[2];
        return predict(x, posteriori);
    }

    /**
     * Majority-vote prediction; fills {@code posteriori[0]} / {@code posteriori[1]}
     * with the vote fractions for class 0 / class 1.
     */
    public int predict(Tuple x, double[] posteriori) {
        int votes1 = 0;
        for (DecisionTree tree : trees) {
            if (tree.predict(x) == 1) {
                votes1++;
            }
        }
        double p1 = (double) votes1 / trees.length;
        posteriori[0] = 1.0 - p1;
        posteriori[1] = p1;
        return p1 >= 0.5 ? 1 : 0;
    }
}
