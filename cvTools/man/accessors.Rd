\name{accessors}
\alias{accessors}
\alias{cvNames}
\alias{cvNames<-}
\alias{fits}
\alias{fits<-}
\alias{ncv}
\alias{nfits}
\title{Access or set information on cross-validation results}
\usage{
  cvNames(x)

  cvNames(x) <- value

  fits(x)

  fits(x) <- value

  ncv(x)

  nfits(x)
}
\arguments{
  \item{x}{an object inheriting from class \code{"cv"} or
  \code{"cvSelect"} that contains cross-validation
  results.}

  \item{value}{a vector of replacement values.}
}
\value{
  \code{cvNames} returns the names of the cross-validation
  results.  The replacement function thereby returns them
  invisibly.

  \code{fits} returns the identifiers of the models for
  objects inheriting from class \code{"cvSelect"} and
  \code{NULL} for objects inheriting from class
  \code{"cv"}.  The replacement function thereby returns
  those values invisibly.

  \code{ncv} returns the number of cross-validation
  results.

  \code{nfits} returns the number of models included in
  objects inheriting from class \code{"cvSelect"} and
  \code{NULL} for objects inheriting from class
  \code{"cv"}.
}
\description{
  Retrieve or set the names of cross-validation results,
  retrieve or set the identifiers of the models, or
  retrieve the number of cross-validation results or
  included models.
}
\examples{
library("robustbase")
data("coleman")
set.seed(1234)  # set seed for reproducibility

## set up folds for cross-validation
folds <- cvFolds(nrow(coleman), K = 5, R = 10)


## compare raw and reweighted LTS estimators for 
## 50\% and 75\% subsets

# 50\% subsets
fitLts50 <- ltsReg(Y ~ ., data = coleman, alpha = 0.5)
cvFitLts50 <- cvLts(fitLts50, cost = rtmspe, folds = folds, 
    fit = "both", trim = 0.1)

# 75\% subsets
fitLts75 <- ltsReg(Y ~ ., data = coleman, alpha = 0.75)
cvFitLts75 <- cvLts(fitLts75, cost = rtmspe, folds = folds, 
    fit = "both", trim = 0.1)

# combine results into one object
cvFitsLts <- cvSelect("0.5" = cvFitLts50, "0.75" = cvFitLts75)
cvFitsLts

# "cv" object
ncv(cvFitLts50)
nfits(cvFitLts50)
cvNames(cvFitLts50)
cvNames(cvFitLts50) <- c("improved", "initial")
fits(cvFitLts50)
cvFitLts50

# "cvSelect" object
ncv(cvFitsLts)
nfits(cvFitsLts)
cvNames(cvFitsLts)
cvNames(cvFitsLts) <- c("improved", "initial")
fits(cvFitsLts)
fits(cvFitsLts) <- 1:2
cvFitsLts
}
\author{
  Andreas Alfons
}
\seealso{
  \code{\link{cvFit}}, \code{\link{cvSelect}},
  \code{\link{cvTuning}}
}
\keyword{utilities}

