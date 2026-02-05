# Questions to answer:

## 1. Reflect on the generated images for each s. How different values of s impact the generation?

![Figure 1](img/varied_CFG_1.png)

1. The image with s=1 looks not bad, but does not represent the input prompt of __"House of a Swiss Family in the Swiss Alps, scenic view, beautiful lighting, ultra detailed, 8k"__ very well.
2. The second image with s=7.5 still adheres to the style of the first photo, showing the inside of the house, insted of the outside like the prompt suggested.
3. Only with the highest guidance scale of s=15 does the prompt result in an image that represents the prompt well.
 

![Figure 2](img/varied_CFG_2.png)
1. Running the same prompt again yielded better results, with s=7.5 already producing the desired output quite closely.

- Conclusion: Test out the guidance scale across a range of values to get the best result, instead of just using a standard parameter, assuming it will work fine.