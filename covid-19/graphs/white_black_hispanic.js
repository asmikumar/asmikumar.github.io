Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

var line_chart = document.getElementById("white_black_hispanic");

var dataFirst = {
    label: "Median White Pay Per Hour",
    data: [17.94, 17.51, 17.44, 17.53, 17.48, 17.71, 17.45, 17.39, 16.98, 17.25,
          17.11, 17.32, 17.27, 17.78, 17.74, 17.83, 17.78, 17.82, 18.09, 17.83,
          17.57, 17.36, 17.64, 17.81, 18.13, 18.77, 19.06, 19.28, 19.85, 20.04,
          20.17, 20.26, 19.77, 19.81, 19.91, 20.03, 20.41, 20.26, 19.86, 19.98,
          19.86, 19.86, 20.42, 20.59, 20.86, 20.94, 21.32],
    lineTension: 0,
    fill: false,
    borderColor: "rgba(0,153,255,1)",
    backgroundColor: "rgba(0,153,255,0.2)"
  };

var dataSecond = {
    label: "Median Black Pay Per Hour",
    data: [13.95, 14.04, 14.11, 14.2, 14.18, 14.15, 14.58, 14.38, 14.01, 13.91,
          13.98, 13.95, 13.93, 14.23, 14.26, 14.55, 14.15, 14.23, 14.09, 14.18,
          14.08, 13.86, 13.97, 13.93, 14.24, 14.95, 15.32, 15.28, 15.36, 15.77,
          16.09, 16.22, 15.61, 15.73, 15.46, 15.41, 16.05, 15.96, 15.46, 15.06,
          15.39, 15.07, 15.19, 15.86, 15.59, 15.59, 16.12],
    lineTension: 0,
    fill: false,
    borderColor: "rgba(102,102,204,1)",
    backgroundColor: "rgba(0102,102,204,0.2)"
  };

  var dataThird = {
      label: "Median Hispanic Pay Per Hour",
      data: [14.3, 14.17, 14.09, 14.16, 14.11, 14.1, 14.39, 14.23, 13.63, 13.75,
            13.62, 13.77, 13.7, 13.69, 13.68, 13.52, 13.27, 13.27, 12.95, 13.06,
            12.95, 12.65, 12.47, 12.82, 12.82, 13.23, 13.43, 13.44, 14.08, 14.18,
            14.01, 13.7, 13.72, 14.04, 14.32, 14.34, 14.44, 14.14, 13.77, 13.75,
            13.74, 14.01, 14.42, 14.99, 15.54, 15.31, 15.89],
      lineTension: 0,
      fill: false,
      borderColor: "rgba(255,204,0,1)",
      backgroundColor: "rgba(255,204,0,0.2)"
    };

var data = {
  labels: ["1973", "1974", "1975", "1976", "1977", "1978", "1979", "1980", "1981", "1982",
          "1983", "1984", "1985", "1986", "1987", "1988", "1989", "1990", "1991", "1992",
          "1993", "1994", "1995", "1996", "1997", "1998", "1999", "2000", "2001", "2002",
          "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012",
          "2013", "2014", "2015", "2016", "2017", "2018", "2019"],
  datasets: [dataFirst, dataSecond, dataThird]
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

var white_black_hispanic = new Chart(line_chart, {
  type: 'line',
  data: data,
  options: chartOptions
});
