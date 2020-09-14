Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

var line_chart = document.getElementById("male_female");

var dataFirst = {
    label: "Median Male Pay Per Hour",
    data: [20.97, 20.71, 21.04, 20.7, 20.9, 21.2, 21.07, 20.87, 20.43, 20.44,
          20.06, 20.2, 20.29, 20.5, 20.57, 20.63, 19.92, 19.16, 19, 19.01, 19.05,
          18.81, 18.91, 18.8, 19.11, 19.62, 20.07, 20.3, 20.44, 20.73, 20.8,
          20.43, 19.97, 20.24, 20.33, 20.31, 20.88, 20.39, 19.89, 19.98, 19.78,
          19.59, 20.26, 20.48, 20.78, 20.47, 21],
    lineTension: 0,
    fill: false,
    borderColor: "rgba(102,204,0,1)",
    backgroundColor: "rgba(102,204,0,0.2)"
  };

var dataSecond = {
    label: "Median Female Pay Per Hour",
    data: [13.19, 13, 13.18, 13.3, 13.2, 13.25, 13.35, 13.29, 13.36, 13.21, 13.45,
          13.69, 13.78, 14.02, 14.2, 14.5, 14.4, 14.37, 14.59, 14.56, 14.82, 14.71,
          14.68, 14.68, 14.96, 15.58, 15.51, 15.75, 16.07, 16.55, 16.71, 16.49, 16.39,
          16.5, 16.55, 16.71, 17.1, 17.01, 16.89, 16.61, 16.48, 16.28, 16.66, 16.99,
          17.2, 17.24, 17.84],
    lineTension: 0,
    fill: false,
    borderColor: "rgba(255,51,0,1)",
    backgroundColor: "rgba(255,51,0,0.2)"
  };



var data = {
  labels: ["1973", "1974", "1975", "1976", "1977", "1978", "1979", "1980", "1981", "1982",
          "1983", "1984", "1985", "1986", "1987", "1988", "1989", "1990", "1991", "1992",
          "1993", "1994", "1995", "1996", "1997", "1998", "1999", "2000", "2001", "2002",
          "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012",
          "2013", "2014", "2015", "2016", "2017", "2018", "2019"],
  datasets: [dataFirst, dataSecond]
};

var chartOptions = {
  legend: {
    display: true,
    position: 'top',
    labels: {
      fontColor: 'black'
    }
  },

};

var male_female = new Chart(line_chart, {
  type: 'line',
  data: data,
  options: chartOptions
});
