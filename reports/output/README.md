# Output Summary
The purpose of this document is to inform the reader how to best utilize the output from the tool. I have provided two different outputs from the tool, that run off the order file from a previous project that was for a fruit packaging facility. The only difference is that one, I included the facility transformation, and the other I did not. This was because There were clear correlations with facilities, but they didn't really generate extra knowledge. IE, we already knew cherries went out of one building and pears another. This kept showing up in the output, so I ran one without that transformation as well.

 Here is a key of the columns it created from the SKU Types
 A- apples <br>
 C- cherries <br>
 M - mixed <br>
 P - pears <br>
 R - Rainier Cherries <br>

All comments will be referring to the "non-facility" run.

If you look at the first result, you'll see it combined three transformations, being UOH(unit of Handling), OCA (Order Commonality), and Seasonality. It only identified two clusters. The output states the first one, cluster 0, was only Apples and pears that were rarely ordered with Cherries, Rainiers, or Mixed SKU's. The second cluster was all cherries, that are almost never ordered with Apples or Pears. You would also notice that these cherries are more commonly ordered in the second and third quarters of the year, and they are ordered less over all.

By looking at the heatmap, we can see that cluster 0 was more commonly ordered with pears and apples, in the first and fourth quarters. We can also see that cluster 1 is more commonly ordered in uoh2 or uoh3 than the average SKU.

You could then continue down through each chosen transformation combination and try to make more meaningful insights.

This is just one example of one takeway from the tool output. We can also look at the feature weights chosen to generate some knowledge. If you average the feature weights and rank them descedning, you can try to find the most polarizing traits of a SKU. In this example, the SKU Types were commonly the easiest to separate into clusters. Order commonality with cherries was also chosen significantly, which shows that is a defining trait of the operation. Things are either ordered with cherries or not, and that can separate your SKU base. 
