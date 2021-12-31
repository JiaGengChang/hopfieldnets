library(dplyr)
library(tidyr)
library(ggbeeswarm)

x0 = read.table('s100_n15_c0.tsv',sep='\t',header = 1)['avg..acc']
x10 = read.table('s100_n15_c10.tsv',sep='\t',header = 1)['avg..acc']
x20 = read.table('s100_n15_c20.tsv',sep='\t',header = 1)['avg..acc']
x30 = read.table('s100_n15_c30.tsv',sep='\t',header = 1)['avg..acc']
x40 = read.table('s100_n15_c40.tsv',sep='\t',header = 1)['avg..acc']
x50 = read.table('s100_n15_c50.tsv',sep='\t',header = 1)['avg..acc']

df = tibble(n=1:50, 
            '0'=x0, 
            '10'=x10, 
            '20'=x20,
            '30'=x30,
            '40'=x40,
            '50'=x50)

p = df %>% 
  pivot_longer(!n, names_to="num. corrupted") %>%
  ggplot(aes(x=`num. corrupted`, y=value$avg..acc, col=`num. corrupted`)) +
  geom_beeswarm(size=0.75) +
  ggtitle("Random corruption of inputs vs accuracy") +
  xlab("Number of corrupted bits") + 
  ylab("Average accuracy") +
  theme(plot.title = element_text(size=20, hjust=0.5),
        axis.title.x = element_text(size=18),
        axis.title.y = element_text(size=18),
        axis.text.x = element_text(size=18),
        axis.text.y = element_text(size=18),
        legend.position="none"
  )
