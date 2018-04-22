
# DataAnalyst Nanodegree Final Project

__ Data Visualization of Prosper Loan Data using Tableau__

__*Karun Gahlawat*__


## Summary
Visualize Prosper Loan data (attached) using Tableau 10.4. Explore data and ask questions as we explore. Create a set of findings and then combine all the findings to create a story to explain the findings. 

## Design

To visualize data we need to first understand the data, variable names, and how they could link to each other intuitively. Once done, we would plot these data individually and with some other relevant factors with the goal to show inter factor behavior. Next step would be to start asking questions towards the end goal of understanding the dataset. Questions like how some variables interact with others and how that interaction changes data behavior. Also along the lines of how the end result was effected by these variables. Next step would be to combine these findings (answers to these questions) into a story. Once we have a story, we would need to show this presentation to other people to get feedback. Once feedback is gathered, we need to make changes to our story or its presentation based on that feedback. Finally wrap it up with final version of our data exploration and its findings and by then hopefully we would have answered all kinds of questions that were brought up during this process.

Leave something else to do at the end in terms of how to improve data and visualization to make it more useful for any academic and / or commercial use

[Here is the workbook ](https://public.tableau.com/profile/karun.gahlawat#!/vizhome/prosper_2/ProsperLoans)


### Some important variables


### Questions as we explore
   1. How does initial number look? Some measures.
      * Sheet InitialNumbers
   1. How do measures span out against term, length of loans in month?
      * Sheet Term
   1. What is count of loans against loanstatus. Various values of loan status could be Cancelled, Chargedoff, Completed, Current, Defaulted, FinalPaymentInProgress, PastDue
      * Sheet Term
   1. How does loanstatus map against borrowers' apr? against term?
      * Sheet BorrowersAPR
   1. What is the distribution of lender's yield? how does it correlate with loan status and term?
      * Sheet LendersYield
   1. What is the estimated loss, filtered on loans after july 2009
      * Sheet EstimatedLoss
   1. What is the estimated return, filtered on loans after july 2009
      * Sheet EstimatedLoss
   1. Could we map estimated loss and estimated return with loan status?
      * Sheet EstimatedLoss
   1. How does prosperscore map against loan status?
      * Sheet ProsperScore
   1. Count of loans against listing category, how do they map with loan status
      * Sheet ListingCategory
   1. Loans by state?
      * Sheet State
   1. Various of occupations? How do they map against loan status?
      * Sheet Occupation
   1. Simple stats on credit lines and open revolving accounts with inquiries and inquiries in last 6 months and bankcardutilization
      * Sheet Credit
   1. Debt to income ratios of borrowers and loan status
      * Sheet Credit
   1. Number of investors that funded the loans. What is the distribution? Did this effect loan status? How does income in these buckets span out?
      * Sheet Investors

## Feedback

   * Listing Category could be shown in text vs number in ListingCategory sheet: Fixed. Created a calculated field and replaced category number with category string. It does make more sense visually now.
   
   * Add Estimated Return on Listing Category sheet: Fixed. Added.
   
   * Include PnL information: Fixed. Added Pnl Sheet.
   
   * Show Percentages i/o Number of Records in Occupation Sheet. Fixed
   
   * Remove some low number categories from PnL sheet. Fixed. Those numbers weren't adding a substantial value and were cluttering space
   
   * Remove low number categories from Investor sheet and add Lenders Yield to compare number of investors vs yield
   
   * Add income range on credit sheet.

## Resources

   1. prosper loan data (attached) from https://www.google.com/url?q=https://s3.amazonaws.com/udacity-hosted-downloads/ud651/prosperLoanData.csv&sa=D&ust=1515287453772000&usg=AFQjCNH9gVD7eanzFg6VIjjDEOE4QJfEKw
   
   1. data dictionary (attached) from  https://www.google.com/url?q=https://docs.google.com/spreadsheet/ccc?key%3D0AllIqIyvWZdadDd5NTlqZ1pBMHlsUjdrOTZHaVBuSlE%26usp%3Dsharing&sa=D&ust=1515287453773000&usg=AFQjCNFrzH6eZAWqBCa8SiFYuNIUb6ObPQ
   
   1. Tableau documentation from https://www.tableau.com/support/help
   

## Next Steps

Use a deep neural net model to classify good loans from bad ones

