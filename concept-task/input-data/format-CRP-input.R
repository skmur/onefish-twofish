library(ggplot2)
library(ggbiplot)
library(dplyr)
library(tidyr)
library(lme4)
library(lmerTest)
library(Rtsne)
library(RColorBrewer)
library(colorRamps)
library(patchwork)
library(ggpubr)
library(scales)
library(lsa)
library(stringr)

getwd()

data <- read.csv("./output-data/CHANGETHIS.csv", header = FALSE) %>%
  filter(!row_number() %in% c(1, 2, 4)) # remove first row

# rename columns
colnames(data) <- c("ID", "Prompting_condition", "Experiment", "Concept", "Choice1", "Choice2", "Response1", "Response2")

### Data formatting
# capital case response columns
data$Response1 <- str_to_title(data$Response1)
data$Response2 <- str_to_title(data$Response2)

# recode responses and factorize
data$Concept <- as.factor(data$Concept)
data$Concept <- recode(data$Concept, " a chicken" = "chicken", " a dolphin" = "dolphin", " a finch" = "finch",
                                     " a penguin" = "penguin", " a robin" = "robin", " a salmon" = "salmon",
                                     " a seal" = "seal", " a whale" = "whale", " an ostrich" = "ostrich",
                                     " an eagle" = "eagle", " Abraham Lincoln" = "Lincoln", " Barack Obama" = "Obama",
                                     " Bernie Sanders" = "Sanders", " Donald Trump" = "Trump", " Elizabeth Warren" = "Warren",
                                     " George W. Bush" = "Bush", " Hillary Clinton" = "Clinton", " Joe Biden" = "Biden",
                                     " Richard Nixon" = "Nixon", " Ronald Reagan" = "Reagan")
# changes the order of the factors for visualization
data$Concept <- factor(data$Concept, levels(data$Concept)[c(3, 5, 1, 10, 11, 4, 6, 7, 2, 8, 9, 12:20)])

# factorize Experiment column (animals vs. politicians)
data$Experiment <- as.factor(data$Experiment)

# make new column for question by combining the two choice options and factorize
data$Question <- paste(as.character(data$Choice1), as.character(data$Choice2))
data$Question <- as.factor(data$Question)

# "We coded each participant’s responses to a single word as a binary vector, corresponding to theforced-choice similarity rating between every other pair of items"
# encode this based on the model's first response
data$ChoiceNumber <- ifelse(as.character(data$Response1) == as.character(data$Choice1), 0, 1)

# If verifying human data, skip above and load in and format data
data <- read.csv("../../input-data/concept-task/Responses.csv", header = FALSE) %>%
  filter(!row_number() %in% c(1, 2, 4)) %>%
  rename_with(~c("i", "Concept", "ID", "Question", "ChoiceNumber"))

### Calculate participant reliability
# .rs.restartR()
tmp <- data %>% 
  group_by(ID, Question, Concept) %>% 
  summarise(Reliability = mean(ChoiceNumber))

tmp$Reliability <- ifelse(tmp$Reliability == 0, 1, ifelse(tmp$Reliability == 1, 1, 0))
data <- merge(data, tmp)

reliability <- data.frame(data$Reliability)
reliability$ID <- data$ID

reliability$lower <- ifelse(data$Reliability,
                            prop.test(sum(data$Reliability), length(data$Reliability))$conf.int[1],
                            prop.test(sum(data$Reliability == 0), length(data$Reliability))$conf.int[1])

reliability$upper <- ifelse(data$Reliability,
                            prop.test(sum(data$Reliability), length(data$Reliability))$conf.int[2],
                            prop.test(sum(data$Reliability == 0), length(data$Reliability))$conf.int[2])

tmp <- reliability %>% group_by(ID) %>% summarise(reliabilityPercent = mean(data.Reliability))
data <- merge(data, tmp)
reliability <- merge(reliability, tmp)

reliability$data.Reliability <- factor(reliability$data.Reliability, levels = c(0, 1), labels = c("Not Reliable", "Reliable"))

### Calculate Intersubject Reliability
tmp <- data %>% group_by(Question, Concept) %>% summarise(QuestionReliability = mean(ChoiceNumber))
tmp <- tmp %>% group_by(Concept) %>% mutate(ConceptReliability = mean(QuestionReliability))
mean(tmp$QuestionReliability)


### Save formatted data
alpha <- .16
probs <- rbeta(1000000, alpha, alpha)
agree <- 1 != rbinom(1000000, 2, probs)
prop.table(table(agree))

mean(data$Reliability) # alpha = .16
mean(subset(data, Experiment %in% c("Animals"))$Reliability) # alpha = .14
mean(subset(data, Experiment %in% c("Politicians"))$Reliability) #alpha = .18

# Write people's responses to CSV for use in the CRP model
responses <- data %>% distinct(ID, Question, Concept, .keep_all = TRUE)
responses <- responses %>% select(Concept, ID, Question, ChoiceNumber)
write.csv(responses, "./output-data/CHANGETHIS.csv")