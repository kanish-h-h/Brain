**Tools - pandas**

*The `pandas` library provides high-performance, easy-to-use data structures and data analysis tools. The main data structure is the `DataFrame`, which you can think of as an in-memory 2D table (like a spreadsheet, with column names and row labels). Many features available in Excel are available programmatically, such as creating pivot tables, computing columns based on other columns, plotting graphs, etc. You can also group rows by column value, or join tables much like in SQL. Pandas is also great at handling time series.*

Prerequisites:
* NumPy – if you are not familiar with NumPy, we recommend that you go through the [NumPy tutorial](tools_numpy.ipynb) now.

<table align="left">
  <td>
    <a href="https://colab.research.google.com/github/ageron/handson-ml3/blob/main/tools_pandas.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  </td>
  <td>
    <a target="_blank" href="https://kaggle.com/kernels/welcome?src=https://github.com/ageron/handson-ml3/blob/main/tools_pandas.ipynb"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>
  </td>
</table>

# Setup

First, let's import `pandas`. People usually import it as `pd`:


```python
import pandas as pd
```

# `Series` objects
The `pandas` library contains the following useful data structures:
* `Series` objects, that we will discuss now. A `Series` object is 1D array, similar to a column in a spreadsheet (with a column name and row labels).
* `DataFrame` objects. This is a 2D table, similar to a spreadsheet (with column names and row labels).
* `Panel` objects. You can see a `Panel` as a dictionary of `DataFrame`s. These are less used, so we will not discuss them here.

## Creating a `Series`
Let's start by creating our first `Series` object!


```python
s = pd.Series([2,-1,3,5])
s
```




    0    2
    1   -1
    2    3
    3    5
    dtype: int64



## Similar to a 1D `ndarray`
`Series` objects behave much like one-dimensional NumPy `ndarray`s, and you can often pass them as parameters to NumPy functions:


```python
import numpy as np
np.exp(s)
```




    0      7.389056
    1      0.367879
    2     20.085537
    3    148.413159
    dtype: float64



Arithmetic operations on `Series` are also possible, and they apply *elementwise*, just like for `ndarray`s:


```python
s + [1000,2000,3000,4000]
```




    0    1002
    1    1999
    2    3003
    3    4005
    dtype: int64



Similar to NumPy, if you add a single number to a `Series`, that number is added to all items in the `Series`. This is called * broadcasting*:


```python
s + 1000
```




    0    1002
    1     999
    2    1003
    3    1005
    dtype: int64



The same is true for all binary operations such as `*` or `/`, and even conditional operations:


```python
s < 0
```




    0    False
    1     True
    2    False
    3    False
    dtype: bool



## Index labels
Each item in a `Series` object has a unique identifier called the *index label*. By default, it is simply the rank of the item in the `Series` (starting from `0`) but you can also set the index labels manually:


```python
s2 = pd.Series([68, 83, 112, 68], index=["alice", "bob", "charles", "darwin"])
s2
```




    alice       68
    bob         83
    charles    112
    darwin      68
    dtype: int64



You can then use the `Series` just like a `dict`:


```python
s2["bob"]
```




    83



You can still access the items by integer location, like in a regular array:


```python
s2[1]
```




    83



To make it clear when you are accessing by label or by integer location, it is recommended to always use the `loc` attribute when accessing by label, and the `iloc` attribute when accessing by integer location:


```python
s2.loc["bob"]
```




    83




```python
s2.iloc[1]
```




    83



Slicing a `Series` also slices the index labels:


```python
s2.iloc[1:3]
```




    bob         83
    charles    112
    dtype: int64



This can lead to unexpected results when using the default numeric labels, so be careful:


```python
surprise = pd.Series([1000, 1001, 1002, 1003])
surprise
```




    0    1000
    1    1001
    2    1002
    3    1003
    dtype: int64




```python
surprise_slice = surprise[2:]
surprise_slice
```




    2    1002
    3    1003
    dtype: int64



Oh, look! The first element has index label `2`. The element with index label `0` is absent from the slice:


```python
try:
    surprise_slice[0]
except KeyError as e:
    print("Key error:", e)
```

    Key error: 0


But remember that you can access elements by integer location using the `iloc` attribute. This illustrates another reason why it's always better to use `loc` and `iloc` to access `Series` objects:


```python
surprise_slice.iloc[0]
```




    1002



## Init from `dict`
You can create a `Series` object from a `dict`. The keys will be used as index labels:


```python
weights = {"alice": 68, "bob": 83, "colin": 86, "darwin": 68}
s3 = pd.Series(weights)
s3
```




    alice     68
    bob       83
    colin     86
    darwin    68
    dtype: int64



You can control which elements you want to include in the `Series` and in what order by explicitly specifying the desired `index`:


```python
s4 = pd.Series(weights, index = ["colin", "alice"])
s4
```




    colin    86
    alice    68
    dtype: int64



## Automatic alignment
When an operation involves multiple `Series` objects, `pandas` automatically aligns items by matching index labels.


```python
print(s2.keys())
print(s3.keys())

s2 + s3
```

    Index(['alice', 'bob', 'charles', 'darwin'], dtype='object')
    Index(['alice', 'bob', 'colin', 'darwin'], dtype='object')





    alice      136.0
    bob        166.0
    charles      NaN
    colin        NaN
    darwin     136.0
    dtype: float64



The resulting `Series` contains the union of index labels from `s2` and `s3`. Since `"colin"` is missing from `s2` and `"charles"` is missing from `s3`, these items have a `NaN` result value (i.e. Not-a-Number means *missing*).

Automatic alignment is very handy when working with data that may come from various sources with varying structure and missing items. But if you forget to set the right index labels, you can have surprising results:


```python
s5 = pd.Series([1000,1000,1000,1000])
print("s2 =", s2.values)
print("s5 =", s5.values)

s2 + s5
```

    s2 = [ 68  83 112  68]
    s5 = [1000 1000 1000 1000]





    alice     NaN
    bob       NaN
    charles   NaN
    darwin    NaN
    0         NaN
    1         NaN
    2         NaN
    3         NaN
    dtype: float64



Pandas could not align the `Series`, since their labels do not match at all, hence the full `NaN` result.

## Init with a scalar
You can also initialize a `Series` object using a scalar and a list of index labels: all items will be set to the scalar.


```python
meaning = pd.Series(42, ["life", "universe", "everything"])
meaning
```




    life          42
    universe      42
    everything    42
    dtype: int64



## `Series` name
A `Series` can have a `name`:


```python
s6 = pd.Series([83, 68], index=["bob", "alice"], name="weights")
s6
```




    bob      83
    alice    68
    Name: weights, dtype: int64



## Plotting a `Series`
Pandas makes it easy to plot `Series` data using matplotlib (for more details on matplotlib, check out the [matplotlib tutorial](tools_matplotlib.ipynb)). Just import matplotlib and call the `plot()` method:


```python
import matplotlib.pyplot as plt
temperatures = [4.4,5.1,6.1,6.2,6.1,6.1,5.7,5.2,4.7,4.1,3.9,3.5]
s7 = pd.Series(temperatures, name="Temperature")
s7.plot()
plt.show()
```


    
![png](PANDAS_48_0.png)
    


There are *many* options for plotting your data. It is not necessary to list them all here: if you need a particular type of plot (histograms, pie charts, etc.), just look for it in the excellent [Visualization](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html) section of pandas' documentation, and look at the example code.

# Handling time
Many datasets have timestamps, and pandas is awesome at manipulating such data:
* it can represent periods (such as 2016Q3) and frequencies (such as "monthly"),
* it can convert periods to actual timestamps, and *vice versa*,
* it can resample data and aggregate values any way you like,
* it can handle timezones.

## Time range
Let's start by creating a time series using `pd.date_range()`. It returns a `DatetimeIndex` containing one datetime per hour for 12 hours starting on October 29th 2016 at 5:30pm.


```python
dates = pd.date_range('2016/10/29 5:30pm', periods=12, freq='H')
dates
```




    DatetimeIndex(['2016-10-29 17:30:00', '2016-10-29 18:30:00',
                   '2016-10-29 19:30:00', '2016-10-29 20:30:00',
                   '2016-10-29 21:30:00', '2016-10-29 22:30:00',
                   '2016-10-29 23:30:00', '2016-10-30 00:30:00',
                   '2016-10-30 01:30:00', '2016-10-30 02:30:00',
                   '2016-10-30 03:30:00', '2016-10-30 04:30:00'],
                  dtype='datetime64[ns]', freq='H')



This `DatetimeIndex` may be used as an index in a `Series`:


```python
temp_series = pd.Series(temperatures, dates)
temp_series
```




    2016-10-29 17:30:00    4.4
    2016-10-29 18:30:00    5.1
    2016-10-29 19:30:00    6.1
    2016-10-29 20:30:00    6.2
    2016-10-29 21:30:00    6.1
    2016-10-29 22:30:00    6.1
    2016-10-29 23:30:00    5.7
    2016-10-30 00:30:00    5.2
    2016-10-30 01:30:00    4.7
    2016-10-30 02:30:00    4.1
    2016-10-30 03:30:00    3.9
    2016-10-30 04:30:00    3.5
    Freq: H, dtype: float64



Let's plot this series:


```python
temp_series.plot(kind="bar")

plt.grid(True)
plt.show()
```


    
![png](PANDAS_55_0.png)
    


## Resampling
Pandas lets us resample a time series very simply. Just call the `resample()` method and specify a new frequency:


```python
temp_series_freq_2H = temp_series.resample("2H")
temp_series_freq_2H
```




    DatetimeIndexResampler [freq=<2 * Hours>, axis=0, closed=left, label=left, convention=start, base=0]



The resampling operation is actually a deferred operation, which is why we did not get a `Series` object, but a `DatetimeIndexResampler` object instead. To actually perform the resampling operation, we can simply call the `mean()` method. Pandas will compute the mean of every pair of consecutive hours:


```python
temp_series_freq_2H = temp_series_freq_2H.mean()
```

Let's plot the result:


```python
temp_series_freq_2H.plot(kind="bar")
plt.show()
```


    
![png](PANDAS_61_0.png)
    


Note how the values have automatically been aggregated into 2-hour periods. If we look at the 6-8pm period, for example, we had a value of `5.1` at 6:30pm, and `6.1` at 7:30pm. After resampling, we just have one value of `5.6`, which is the mean of `5.1` and `6.1`. Rather than computing the mean, we could have used any other aggregation function, for example we can decide to keep the minimum value of each period:


```python
temp_series_freq_2H = temp_series.resample("2H").min()
temp_series_freq_2H
```




    2016-10-29 16:00:00    4.4
    2016-10-29 18:00:00    5.1
    2016-10-29 20:00:00    6.1
    2016-10-29 22:00:00    5.7
    2016-10-30 00:00:00    4.7
    2016-10-30 02:00:00    3.9
    2016-10-30 04:00:00    3.5
    Freq: 2H, dtype: float64



Or, equivalently, we could use the `apply()` method instead:


```python
temp_series_freq_2H = temp_series.resample("2H").apply(np.min)
temp_series_freq_2H
```




    2016-10-29 16:00:00    4.4
    2016-10-29 18:00:00    5.1
    2016-10-29 20:00:00    6.1
    2016-10-29 22:00:00    5.7
    2016-10-30 00:00:00    4.7
    2016-10-30 02:00:00    3.9
    2016-10-30 04:00:00    3.5
    Freq: 2H, dtype: float64



## Upsampling and interpolation
It was an example of downsampling. We can also upsample (i.e. increase the frequency), but it will create holes in our data:


```python
temp_series_freq_15min = temp_series.resample("15Min").mean()
temp_series_freq_15min.head(n=10) # `head` displays the top n values
```




    2016-10-29 17:30:00    4.4
    2016-10-29 17:45:00    NaN
    2016-10-29 18:00:00    NaN
    2016-10-29 18:15:00    NaN
    2016-10-29 18:30:00    5.1
    2016-10-29 18:45:00    NaN
    2016-10-29 19:00:00    NaN
    2016-10-29 19:15:00    NaN
    2016-10-29 19:30:00    6.1
    2016-10-29 19:45:00    NaN
    Freq: 15T, dtype: float64



One solution is to fill the gaps by interpolating. We just call the `interpolate()` method. The default is to use linear interpolation, but we can also select another method, such as cubic interpolation:


```python
temp_series_freq_15min = temp_series.resample("15Min").interpolate(method="cubic")
temp_series_freq_15min.head(n=10)
```




    2016-10-29 17:30:00    4.400000
    2016-10-29 17:45:00    4.452911
    2016-10-29 18:00:00    4.605113
    2016-10-29 18:15:00    4.829758
    2016-10-29 18:30:00    5.100000
    2016-10-29 18:45:00    5.388992
    2016-10-29 19:00:00    5.669887
    2016-10-29 19:15:00    5.915839
    2016-10-29 19:30:00    6.100000
    2016-10-29 19:45:00    6.203621
    Freq: 15T, dtype: float64




```python
temp_series.plot(label="Period: 1 hour")
temp_series_freq_15min.plot(label="Period: 15 minutes")
plt.legend()
plt.show()
```


    
![png](PANDAS_70_0.png)
    


## Timezones
By default, datetimes are *naive*: they are not aware of timezones, so 2016-10-30 02:30 might mean October 30th 2016 at 2:30am in Paris or in New York. We can make datetimes timezone *aware* by calling the `tz_localize()` method:


```python
temp_series_ny = temp_series.tz_localize("America/New_York")
temp_series_ny
```




    2016-10-29 17:30:00-04:00    4.4
    2016-10-29 18:30:00-04:00    5.1
    2016-10-29 19:30:00-04:00    6.1
    2016-10-29 20:30:00-04:00    6.2
    2016-10-29 21:30:00-04:00    6.1
    2016-10-29 22:30:00-04:00    6.1
    2016-10-29 23:30:00-04:00    5.7
    2016-10-30 00:30:00-04:00    5.2
    2016-10-30 01:30:00-04:00    4.7
    2016-10-30 02:30:00-04:00    4.1
    2016-10-30 03:30:00-04:00    3.9
    2016-10-30 04:30:00-04:00    3.5
    Freq: H, dtype: float64



Note that `-04:00` is now appended to all the datetimes. It means that these datetimes refer to [UTC](https://en.wikipedia.org/wiki/Coordinated_Universal_Time) - 4 hours.

We can convert these datetimes to Paris time like this:


```python
temp_series_paris = temp_series_ny.tz_convert("Europe/Paris")
temp_series_paris
```




    2016-10-29 23:30:00+02:00    4.4
    2016-10-30 00:30:00+02:00    5.1
    2016-10-30 01:30:00+02:00    6.1
    2016-10-30 02:30:00+02:00    6.2
    2016-10-30 02:30:00+01:00    6.1
    2016-10-30 03:30:00+01:00    6.1
    2016-10-30 04:30:00+01:00    5.7
    2016-10-30 05:30:00+01:00    5.2
    2016-10-30 06:30:00+01:00    4.7
    2016-10-30 07:30:00+01:00    4.1
    2016-10-30 08:30:00+01:00    3.9
    2016-10-30 09:30:00+01:00    3.5
    Freq: H, dtype: float64



You may have noticed that the UTC offset changes from `+02:00` to `+01:00`: this is because France switches to winter time at 3am that particular night (time goes back to 2am). Notice that 2:30am occurs twice! Let's go back to a naive representation (if you log some data hourly using local time, without storing the timezone, you might get something like this):


```python
temp_series_paris_naive = temp_series_paris.tz_localize(None)
temp_series_paris_naive
```




    2016-10-29 23:30:00    4.4
    2016-10-30 00:30:00    5.1
    2016-10-30 01:30:00    6.1
    2016-10-30 02:30:00    6.2
    2016-10-30 02:30:00    6.1
    2016-10-30 03:30:00    6.1
    2016-10-30 04:30:00    5.7
    2016-10-30 05:30:00    5.2
    2016-10-30 06:30:00    4.7
    2016-10-30 07:30:00    4.1
    2016-10-30 08:30:00    3.9
    2016-10-30 09:30:00    3.5
    Freq: H, dtype: float64



Now `02:30` is really ambiguous. If we try to localize these naive datetimes to the Paris timezone, we get an error:


```python
try:
    temp_series_paris_naive.tz_localize("Europe/Paris")
except Exception as e:
    print(type(e))
    print(e)
```

    <class 'pytz.exceptions.AmbiguousTimeError'>
    Cannot infer dst time from Timestamp('2016-10-30 02:30:00'), try using the 'ambiguous' argument


Fortunately, by using the `ambiguous` argument we can tell pandas to infer the right DST (Daylight Saving Time) based on the order of the ambiguous timestamps:


```python
temp_series_paris_naive.tz_localize("Europe/Paris", ambiguous="infer")
```




    2016-10-29 23:30:00+02:00    4.4
    2016-10-30 00:30:00+02:00    5.1
    2016-10-30 01:30:00+02:00    6.1
    2016-10-30 02:30:00+02:00    6.2
    2016-10-30 02:30:00+01:00    6.1
    2016-10-30 03:30:00+01:00    6.1
    2016-10-30 04:30:00+01:00    5.7
    2016-10-30 05:30:00+01:00    5.2
    2016-10-30 06:30:00+01:00    4.7
    2016-10-30 07:30:00+01:00    4.1
    2016-10-30 08:30:00+01:00    3.9
    2016-10-30 09:30:00+01:00    3.5
    Freq: H, dtype: float64



## Periods
The `pd.period_range()` function returns a `PeriodIndex` instead of a `DatetimeIndex`. For example, let's get all quarters in 2016 and 2017:


```python
quarters = pd.period_range('2016Q1', periods=8, freq='Q')
quarters
```




    PeriodIndex(['2016Q1', '2016Q2', '2016Q3', '2016Q4', '2017Q1', '2017Q2',
                 '2017Q3', '2017Q4'],
                dtype='period[Q-DEC]', freq='Q-DEC')



Adding a number `N` to a `PeriodIndex` shifts the periods by `N` times the `PeriodIndex`'s frequency:


```python
quarters + 3
```




    PeriodIndex(['2016Q4', '2017Q1', '2017Q2', '2017Q3', '2017Q4', '2018Q1',
                 '2018Q2', '2018Q3'],
                dtype='period[Q-DEC]', freq='Q-DEC')



The `asfreq()` method lets us change the frequency of the `PeriodIndex`. All periods are lengthened or shortened accordingly. For example, let's convert all the quarterly periods to monthly periods (zooming in):


```python
quarters.asfreq("M")
```




    PeriodIndex(['2016-03', '2016-06', '2016-09', '2016-12', '2017-03', '2017-06',
                 '2017-09', '2017-12'],
                dtype='period[M]', freq='M')



By default, the `asfreq` zooms on the end of each period. We can tell it to zoom on the start of each period instead:


```python
quarters.asfreq("M", how="start")
```




    PeriodIndex(['2016-01', '2016-04', '2016-07', '2016-10', '2017-01', '2017-04',
                 '2017-07', '2017-10'],
                dtype='period[M]', freq='M')



And we can zoom out:


```python
quarters.asfreq("A")
```




    PeriodIndex(['2016', '2016', '2016', '2016', '2017', '2017', '2017', '2017'], dtype='period[A-DEC]', freq='A-DEC')



Of course, we can create a `Series` with a `PeriodIndex`:


```python
quarterly_revenue = pd.Series([300, 320, 290, 390, 320, 360, 310, 410], index=quarters)
quarterly_revenue
```




    2016Q1    300
    2016Q2    320
    2016Q3    290
    2016Q4    390
    2017Q1    320
    2017Q2    360
    2017Q3    310
    2017Q4    410
    Freq: Q-DEC, dtype: int64




```python
quarterly_revenue.plot(kind="line")
plt.show()
```


    
![png](PANDAS_93_0.png)
    


We can convert periods to timestamps by calling `to_timestamp`. By default, it will give us the first day of each period, but by setting `how` and `freq`, we can get the last hour of each period:


```python
last_hours = quarterly_revenue.to_timestamp(how="end", freq="H")
last_hours
```




    2016-03-31 23:00:00    300
    2016-06-30 23:00:00    320
    2016-09-30 23:00:00    290
    2016-12-31 23:00:00    390
    2017-03-31 23:00:00    320
    2017-06-30 23:00:00    360
    2017-09-30 23:00:00    310
    2017-12-31 23:00:00    410
    Freq: Q-DEC, dtype: int64



And back to periods by calling `to_period`:


```python
last_hours.to_period()
```




    2016Q1    300
    2016Q2    320
    2016Q3    290
    2016Q4    390
    2017Q1    320
    2017Q2    360
    2017Q3    310
    2017Q4    410
    Freq: Q-DEC, dtype: int64



Pandas also provides many other time-related functions that we recommend you check out in the [documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html). To whet your appetite, here is one way to get the last business day of each month in 2016, at 9am:


```python
months_2016 = pd.period_range("2016", periods=12, freq="M")
one_day_after_last_days = months_2016.asfreq("D") + 1
last_bdays = one_day_after_last_days.to_timestamp() - pd.tseries.offsets.BDay()
last_bdays.to_period("H") + 9
```




    PeriodIndex(['2016-01-29 09:00', '2016-02-29 09:00', '2016-03-31 09:00',
                 '2016-04-29 09:00', '2016-05-31 09:00', '2016-06-30 09:00',
                 '2016-07-29 09:00', '2016-08-31 09:00', '2016-09-30 09:00',
                 '2016-10-31 09:00', '2016-11-30 09:00', '2016-12-30 09:00'],
                dtype='period[H]', freq='H')



# `DataFrame` objects
A DataFrame object represents a spreadsheet, with cell values, column names and row index labels. You can define expressions to compute columns based on other columns, create pivot-tables, group rows, draw graphs, etc. You can see `DataFrame`s as dictionaries of `Series`.

## Creating a `DataFrame`
You can create a DataFrame by passing a dictionary of `Series` objects:


```python
people_dict = {
    "weight": pd.Series([68, 83, 112], index=["alice", "bob", "charles"]),
    "birthyear": pd.Series([1984, 1985, 1992], index=["bob", "alice", "charles"], name="year"),
    "children": pd.Series([0, 3], index=["charles", "bob"]),
    "hobby": pd.Series(["Biking", "Dancing"], index=["alice", "bob"]),
}
people = pd.DataFrame(people_dict)
people
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>birthyear</th>
      <th>children</th>
      <th>hobby</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>1985</td>
      <td>NaN</td>
      <td>Biking</td>
      <td>68</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>1984</td>
      <td>3.0</td>
      <td>Dancing</td>
      <td>83</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>1992</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>112</td>
    </tr>
  </tbody>
</table>
</div>



A few things to note:
* the `Series` were automatically aligned based on their index,
* missing values are represented as `NaN`,
* `Series` names are ignored (the name `"year"` was dropped),
* `DataFrame`s are displayed nicely in Jupyter notebooks, woohoo!

You can access columns pretty much as you would expect. They are returned as `Series` objects:


```python
people["birthyear"]
```




    alice      1985
    bob        1984
    charles    1992
    Name: birthyear, dtype: int64



You can also get multiple columns at once:


```python
people[["birthyear", "hobby"]]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>birthyear</th>
      <th>hobby</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>1985</td>
      <td>Biking</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>1984</td>
      <td>Dancing</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>1992</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



If you pass a list of columns and/or index row labels to the `DataFrame` constructor, it will guarantee that these columns and/or rows will exist, in that order, and no other column/row will exist. For example:


```python
d2 = pd.DataFrame(
        people_dict,
        columns=["birthyear", "weight", "height"],
        index=["bob", "alice", "eugene"]
     )
d2
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>birthyear</th>
      <th>weight</th>
      <th>height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bob</th>
      <td>1984.0</td>
      <td>83.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>alice</th>
      <td>1985.0</td>
      <td>68.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>eugene</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Another convenient way to create a `DataFrame` is to pass all the values to the constructor as an `ndarray`, or a list of lists, and specify the column names and row index labels separately:


```python
values = [
            [1985, np.nan, "Biking",   68],
            [1984, 3,      "Dancing",  83],
            [1992, 0,      np.nan,    112]
         ]
d3 = pd.DataFrame(
        values,
        columns=["birthyear", "children", "hobby", "weight"],
        index=["alice", "bob", "charles"]
     )
d3
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>birthyear</th>
      <th>children</th>
      <th>hobby</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>1985</td>
      <td>NaN</td>
      <td>Biking</td>
      <td>68</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>1984</td>
      <td>3.0</td>
      <td>Dancing</td>
      <td>83</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>1992</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>112</td>
    </tr>
  </tbody>
</table>
</div>



To specify missing values, you can use either `np.nan` or NumPy's masked arrays:


```python
masked_array = np.ma.asarray(values, dtype=object)
masked_array[(0, 2), (1, 2)] = np.ma.masked
d3 = pd.DataFrame(
        masked_array,
        columns=["birthyear", "children", "hobby", "weight"],
        index=["alice", "bob", "charles"]
     )
d3
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>birthyear</th>
      <th>children</th>
      <th>hobby</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>1985</td>
      <td>NaN</td>
      <td>Biking</td>
      <td>68</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>1984</td>
      <td>3</td>
      <td>Dancing</td>
      <td>83</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>1992</td>
      <td>0</td>
      <td>NaN</td>
      <td>112</td>
    </tr>
  </tbody>
</table>
</div>



Instead of an `ndarray`, you can also pass a `DataFrame` object:


```python
d4 = pd.DataFrame(
         d3,
         columns=["hobby", "children"],
         index=["alice", "bob"]
     )
d4
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hobby</th>
      <th>children</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>Biking</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>Dancing</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



It is also possible to create a `DataFrame` with a dictionary (or list) of dictionaries (or lists):


```python
people = pd.DataFrame({
    "birthyear": {"alice": 1985, "bob": 1984, "charles": 1992},
    "hobby": {"alice": "Biking", "bob": "Dancing"},
    "weight": {"alice": 68, "bob": 83, "charles": 112},
    "children": {"bob": 3, "charles": 0}
})
people
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>birthyear</th>
      <th>children</th>
      <th>hobby</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>1985</td>
      <td>NaN</td>
      <td>Biking</td>
      <td>68</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>1984</td>
      <td>3.0</td>
      <td>Dancing</td>
      <td>83</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>1992</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>112</td>
    </tr>
  </tbody>
</table>
</div>



## Multi-indexing
If all columns are tuples of the same size, then they are understood as a multi-index. The same goes for row index labels. For example:


```python
d5 = pd.DataFrame(
  {
    ("public", "birthyear"):
        {("Paris","alice"): 1985, ("Paris","bob"): 1984, ("London","charles"): 1992},
    ("public", "hobby"):
        {("Paris","alice"): "Biking", ("Paris","bob"): "Dancing"},
    ("private", "weight"):
        {("Paris","alice"): 68, ("Paris","bob"): 83, ("London","charles"): 112},
    ("private", "children"):
        {("Paris", "alice"): np.nan, ("Paris","bob"): 3, ("London","charles"): 0}
  }
)
d5
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">private</th>
      <th colspan="2" halign="left">public</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>children</th>
      <th>weight</th>
      <th>birthyear</th>
      <th>hobby</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>London</th>
      <th>charles</th>
      <td>0.0</td>
      <td>112</td>
      <td>1992</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Paris</th>
      <th>alice</th>
      <td>NaN</td>
      <td>68</td>
      <td>1985</td>
      <td>Biking</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>3.0</td>
      <td>83</td>
      <td>1984</td>
      <td>Dancing</td>
    </tr>
  </tbody>
</table>
</div>



You can now get a `DataFrame` containing all the `"public"` columns very simply:


```python
d5["public"]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>birthyear</th>
      <th>hobby</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>London</th>
      <th>charles</th>
      <td>1992</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Paris</th>
      <th>alice</th>
      <td>1985</td>
      <td>Biking</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>1984</td>
      <td>Dancing</td>
    </tr>
  </tbody>
</table>
</div>




```python
d5["public", "hobby"]  # Same result as d5["public"]["hobby"]
```




    London  charles        NaN
    Paris   alice       Biking
            bob        Dancing
    Name: (public, hobby), dtype: object



## Dropping a level
Let's look at `d5` again:


```python
d5
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">private</th>
      <th colspan="2" halign="left">public</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>children</th>
      <th>weight</th>
      <th>birthyear</th>
      <th>hobby</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>London</th>
      <th>charles</th>
      <td>0.0</td>
      <td>112</td>
      <td>1992</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Paris</th>
      <th>alice</th>
      <td>NaN</td>
      <td>68</td>
      <td>1985</td>
      <td>Biking</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>3.0</td>
      <td>83</td>
      <td>1984</td>
      <td>Dancing</td>
    </tr>
  </tbody>
</table>
</div>



There are two levels of columns, and two levels of indices. We can drop a column level by calling `droplevel()` (the same goes for indices):


```python
d5.columns = d5.columns.droplevel(level = 0)
d5
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>children</th>
      <th>weight</th>
      <th>birthyear</th>
      <th>hobby</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>London</th>
      <th>charles</th>
      <td>0.0</td>
      <td>112</td>
      <td>1992</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Paris</th>
      <th>alice</th>
      <td>NaN</td>
      <td>68</td>
      <td>1985</td>
      <td>Biking</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>3.0</td>
      <td>83</td>
      <td>1984</td>
      <td>Dancing</td>
    </tr>
  </tbody>
</table>
</div>



## Transposing
You can swap columns and indices using the `T` attribute:


```python
d6 = d5.T
d6
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>London</th>
      <th colspan="2" halign="left">Paris</th>
    </tr>
    <tr>
      <th></th>
      <th>charles</th>
      <th>alice</th>
      <th>bob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>children</th>
      <td>0</td>
      <td>NaN</td>
      <td>3</td>
    </tr>
    <tr>
      <th>weight</th>
      <td>112</td>
      <td>68</td>
      <td>83</td>
    </tr>
    <tr>
      <th>birthyear</th>
      <td>1992</td>
      <td>1985</td>
      <td>1984</td>
    </tr>
    <tr>
      <th>hobby</th>
      <td>NaN</td>
      <td>Biking</td>
      <td>Dancing</td>
    </tr>
  </tbody>
</table>
</div>



## Stacking and unstacking levels
Calling the `stack()` method will push the lowest column level after the lowest index:


```python
d7 = d6.stack()
d7
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>London</th>
      <th>Paris</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">children</th>
      <th>bob</th>
      <td>NaN</td>
      <td>3</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">weight</th>
      <th>alice</th>
      <td>NaN</td>
      <td>68</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>NaN</td>
      <td>83</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>112</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">birthyear</th>
      <th>alice</th>
      <td>NaN</td>
      <td>1985</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>NaN</td>
      <td>1984</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>1992</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">hobby</th>
      <th>alice</th>
      <td>NaN</td>
      <td>Biking</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>NaN</td>
      <td>Dancing</td>
    </tr>
  </tbody>
</table>
</div>



Note that many `NaN` values appeared. This makes sense because many new combinations did not exist before (e.g. there was no `bob` in `London`).

Calling `unstack()` will do the reverse, once again creating many `NaN` values.


```python
d8 = d7.unstack()
d8
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">London</th>
      <th colspan="3" halign="left">Paris</th>
    </tr>
    <tr>
      <th></th>
      <th>alice</th>
      <th>bob</th>
      <th>charles</th>
      <th>alice</th>
      <th>bob</th>
      <th>charles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>children</th>
      <td>None</td>
      <td>NaN</td>
      <td>0</td>
      <td>None</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>weight</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>112</td>
      <td>68</td>
      <td>83</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>birthyear</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1992</td>
      <td>1985</td>
      <td>1984</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>hobby</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>Biking</td>
      <td>Dancing</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



If we call `unstack` again, we end up with a `Series` object:


```python
d9 = d8.unstack()
d9
```




    London  alice    children        None
                     weight           NaN
                     birthyear        NaN
                     hobby            NaN
            bob      children         NaN
                     weight           NaN
                     birthyear        NaN
                     hobby            NaN
            charles  children           0
                     weight           112
                     birthyear       1992
                     hobby           None
    Paris   alice    children        None
                     weight            68
                     birthyear       1985
                     hobby         Biking
            bob      children           3
                     weight            83
                     birthyear       1984
                     hobby        Dancing
            charles  children         NaN
                     weight           NaN
                     birthyear        NaN
                     hobby           None
    dtype: object



The `stack()` and `unstack()` methods let you select the `level` to stack/unstack. You can even stack/unstack multiple levels at once:


```python
d10 = d9.unstack(level = (0,1))
d10
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">London</th>
      <th colspan="3" halign="left">Paris</th>
    </tr>
    <tr>
      <th></th>
      <th>alice</th>
      <th>bob</th>
      <th>charles</th>
      <th>alice</th>
      <th>bob</th>
      <th>charles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>children</th>
      <td>None</td>
      <td>NaN</td>
      <td>0</td>
      <td>None</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>weight</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>112</td>
      <td>68</td>
      <td>83</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>birthyear</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1992</td>
      <td>1985</td>
      <td>1984</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>hobby</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>Biking</td>
      <td>Dancing</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



## Most methods return modified copies
As you may have noticed, the `stack()` and `unstack()` methods do not modify the object they are called on. Instead, they work on a copy and return that copy. This is true of most methods in pandas.

## Accessing rows
Let's go back to the `people` `DataFrame`:


```python
people
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>birthyear</th>
      <th>children</th>
      <th>hobby</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>1985</td>
      <td>NaN</td>
      <td>Biking</td>
      <td>68</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>1984</td>
      <td>3.0</td>
      <td>Dancing</td>
      <td>83</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>1992</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>112</td>
    </tr>
  </tbody>
</table>
</div>



The `loc` attribute lets you access rows instead of columns. The result is a `Series` object in which the `DataFrame`'s column names are mapped to row index labels:


```python
people.loc["charles"]
```




    birthyear    1992
    children        0
    hobby         NaN
    weight        112
    Name: charles, dtype: object



You can also access rows by integer location using the `iloc` attribute:


```python
people.iloc[2]
```




    birthyear    1992
    children        0
    hobby         NaN
    weight        112
    Name: charles, dtype: object



You can also get a slice of rows, and this returns a `DataFrame` object:


```python
people.iloc[1:3]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>birthyear</th>
      <th>children</th>
      <th>hobby</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bob</th>
      <td>1984</td>
      <td>3.0</td>
      <td>Dancing</td>
      <td>83</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>1992</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>112</td>
    </tr>
  </tbody>
</table>
</div>



Finally, you can pass a boolean array to get the matching rows:


```python
people[np.array([True, False, True])]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>birthyear</th>
      <th>children</th>
      <th>hobby</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>1985</td>
      <td>NaN</td>
      <td>Biking</td>
      <td>68</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>1992</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>112</td>
    </tr>
  </tbody>
</table>
</div>



This is most useful when combined with boolean expressions:


```python
people[people["birthyear"] < 1990]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>birthyear</th>
      <th>children</th>
      <th>hobby</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>1985</td>
      <td>NaN</td>
      <td>Biking</td>
      <td>68</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>1984</td>
      <td>3.0</td>
      <td>Dancing</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>



## Adding and removing columns
You can generally treat `DataFrame` objects like dictionaries of `Series`, so the following works fine:


```python
people
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>birthyear</th>
      <th>children</th>
      <th>hobby</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>1985</td>
      <td>NaN</td>
      <td>Biking</td>
      <td>68</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>1984</td>
      <td>3.0</td>
      <td>Dancing</td>
      <td>83</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>1992</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>112</td>
    </tr>
  </tbody>
</table>
</div>




```python
people["age"] = 2018 - people["birthyear"]  # adds a new column "age"
people["over 30"] = people["age"] > 30      # adds another column "over 30"
birthyears = people.pop("birthyear")
del people["children"]

people
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hobby</th>
      <th>weight</th>
      <th>age</th>
      <th>over 30</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>Biking</td>
      <td>68</td>
      <td>33</td>
      <td>True</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>Dancing</td>
      <td>83</td>
      <td>34</td>
      <td>True</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>NaN</td>
      <td>112</td>
      <td>26</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
birthyears
```




    alice      1985
    bob        1984
    charles    1992
    Name: birthyear, dtype: int64



When you add a new column, it must have the same number of rows. Missing rows are filled with NaN, and extra rows are ignored:


```python
people["pets"] = pd.Series({"bob": 0, "charles": 5, "eugene": 1})  # alice is missing, eugene is ignored
people
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hobby</th>
      <th>weight</th>
      <th>age</th>
      <th>over 30</th>
      <th>pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>Biking</td>
      <td>68</td>
      <td>33</td>
      <td>True</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>Dancing</td>
      <td>83</td>
      <td>34</td>
      <td>True</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>NaN</td>
      <td>112</td>
      <td>26</td>
      <td>False</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



When adding a new column, it is added at the end (on the right) by default. You can also insert a column anywhere else using the `insert()` method:


```python
people.insert(1, "height", [172, 181, 185])
people
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hobby</th>
      <th>height</th>
      <th>weight</th>
      <th>age</th>
      <th>over 30</th>
      <th>pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>Biking</td>
      <td>172</td>
      <td>68</td>
      <td>33</td>
      <td>True</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>Dancing</td>
      <td>181</td>
      <td>83</td>
      <td>34</td>
      <td>True</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>NaN</td>
      <td>185</td>
      <td>112</td>
      <td>26</td>
      <td>False</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



## Assigning new columns
You can also create new columns by calling the `assign()` method. Note that this returns a new `DataFrame` object, the original is not modified:


```python
people.assign(
    body_mass_index = people["weight"] / (people["height"] / 100) ** 2,
    has_pets = people["pets"] > 0
)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hobby</th>
      <th>height</th>
      <th>weight</th>
      <th>age</th>
      <th>over 30</th>
      <th>pets</th>
      <th>body_mass_index</th>
      <th>has_pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>Biking</td>
      <td>172</td>
      <td>68</td>
      <td>33</td>
      <td>True</td>
      <td>NaN</td>
      <td>22.985398</td>
      <td>False</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>Dancing</td>
      <td>181</td>
      <td>83</td>
      <td>34</td>
      <td>True</td>
      <td>0.0</td>
      <td>25.335002</td>
      <td>False</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>NaN</td>
      <td>185</td>
      <td>112</td>
      <td>26</td>
      <td>False</td>
      <td>5.0</td>
      <td>32.724617</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



Note that you cannot access columns created within the same assignment:


```python
try:
    people.assign(
        body_mass_index = people["weight"] / (people["height"] / 100) ** 2,
        overweight = people["body_mass_index"] > 25
    )
except KeyError as e:
    print("Key error:", e)
```

    Key error: 'body_mass_index'


The solution is to split this assignment in two consecutive assignments:


```python
d6 = people.assign(body_mass_index = people["weight"] / (people["height"] / 100) ** 2)
d6.assign(overweight = d6["body_mass_index"] > 25)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hobby</th>
      <th>height</th>
      <th>weight</th>
      <th>age</th>
      <th>over 30</th>
      <th>pets</th>
      <th>body_mass_index</th>
      <th>overweight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>Biking</td>
      <td>172</td>
      <td>68</td>
      <td>33</td>
      <td>True</td>
      <td>NaN</td>
      <td>22.985398</td>
      <td>False</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>Dancing</td>
      <td>181</td>
      <td>83</td>
      <td>34</td>
      <td>True</td>
      <td>0.0</td>
      <td>25.335002</td>
      <td>True</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>NaN</td>
      <td>185</td>
      <td>112</td>
      <td>26</td>
      <td>False</td>
      <td>5.0</td>
      <td>32.724617</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



Having to create a temporary variable `d6` is not very convenient. You may want to just chain the assignment calls, but it does not work because the `people` object is not actually modified by the first assignment:


```python
try:
    (people
         .assign(body_mass_index = people["weight"] / (people["height"] / 100) ** 2)
         .assign(overweight = people["body_mass_index"] > 25)
    )
except KeyError as e:
    print("Key error:", e)
```

    Key error: 'body_mass_index'


But fear not, there is a simple solution. You can pass a function to the `assign()` method (typically a `lambda` function), and this function will be called with the `DataFrame` as a parameter:


```python
(people
     .assign(body_mass_index = lambda df: df["weight"] / (df["height"] / 100) ** 2)
     .assign(overweight = lambda df: df["body_mass_index"] > 25)
)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hobby</th>
      <th>height</th>
      <th>weight</th>
      <th>age</th>
      <th>over 30</th>
      <th>pets</th>
      <th>body_mass_index</th>
      <th>overweight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>Biking</td>
      <td>172</td>
      <td>68</td>
      <td>33</td>
      <td>True</td>
      <td>NaN</td>
      <td>22.985398</td>
      <td>False</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>Dancing</td>
      <td>181</td>
      <td>83</td>
      <td>34</td>
      <td>True</td>
      <td>0.0</td>
      <td>25.335002</td>
      <td>True</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>NaN</td>
      <td>185</td>
      <td>112</td>
      <td>26</td>
      <td>False</td>
      <td>5.0</td>
      <td>32.724617</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



Problem solved!

## Evaluating an expression
A great feature supported by pandas is expression evaluation. It relies on the `numexpr` library which must be installed.


```python
people.eval("weight / (height/100) ** 2 > 25")
```




    alice      False
    bob         True
    charles     True
    dtype: bool



Assignment expressions are also supported. Let's set `inplace=True` to directly modify the `DataFrame` rather than getting a modified copy:


```python
people.eval("body_mass_index = weight / (height/100) ** 2", inplace=True)
people
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hobby</th>
      <th>height</th>
      <th>weight</th>
      <th>age</th>
      <th>over 30</th>
      <th>pets</th>
      <th>body_mass_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>Biking</td>
      <td>172</td>
      <td>68</td>
      <td>33</td>
      <td>True</td>
      <td>NaN</td>
      <td>22.985398</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>Dancing</td>
      <td>181</td>
      <td>83</td>
      <td>34</td>
      <td>True</td>
      <td>0.0</td>
      <td>25.335002</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>NaN</td>
      <td>185</td>
      <td>112</td>
      <td>26</td>
      <td>False</td>
      <td>5.0</td>
      <td>32.724617</td>
    </tr>
  </tbody>
</table>
</div>



You can use a local or global variable in an expression by prefixing it with `'@'`:


```python
overweight_threshold = 30
people.eval("overweight = body_mass_index > @overweight_threshold", inplace=True)
people
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hobby</th>
      <th>height</th>
      <th>weight</th>
      <th>age</th>
      <th>over 30</th>
      <th>pets</th>
      <th>body_mass_index</th>
      <th>overweight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>Biking</td>
      <td>172</td>
      <td>68</td>
      <td>33</td>
      <td>True</td>
      <td>NaN</td>
      <td>22.985398</td>
      <td>False</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>Dancing</td>
      <td>181</td>
      <td>83</td>
      <td>34</td>
      <td>True</td>
      <td>0.0</td>
      <td>25.335002</td>
      <td>False</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>NaN</td>
      <td>185</td>
      <td>112</td>
      <td>26</td>
      <td>False</td>
      <td>5.0</td>
      <td>32.724617</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



## Querying a `DataFrame`
The `query()` method lets you filter a `DataFrame` based on a query expression:


```python
people.query("age > 30 and pets == 0")
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hobby</th>
      <th>height</th>
      <th>weight</th>
      <th>age</th>
      <th>over 30</th>
      <th>pets</th>
      <th>body_mass_index</th>
      <th>overweight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bob</th>
      <td>Dancing</td>
      <td>181</td>
      <td>83</td>
      <td>34</td>
      <td>True</td>
      <td>0.0</td>
      <td>25.335002</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



## Sorting a `DataFrame`
You can sort a `DataFrame` by calling its `sort_index` method. By default, it sorts the rows by their index label, in ascending order, but let's reverse the order:


```python
people.sort_index(ascending=False)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hobby</th>
      <th>height</th>
      <th>weight</th>
      <th>age</th>
      <th>over 30</th>
      <th>pets</th>
      <th>body_mass_index</th>
      <th>overweight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>charles</th>
      <td>NaN</td>
      <td>185</td>
      <td>112</td>
      <td>26</td>
      <td>False</td>
      <td>5.0</td>
      <td>32.724617</td>
      <td>True</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>Dancing</td>
      <td>181</td>
      <td>83</td>
      <td>34</td>
      <td>True</td>
      <td>0.0</td>
      <td>25.335002</td>
      <td>False</td>
    </tr>
    <tr>
      <th>alice</th>
      <td>Biking</td>
      <td>172</td>
      <td>68</td>
      <td>33</td>
      <td>True</td>
      <td>NaN</td>
      <td>22.985398</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



Note that `sort_index` returned a sorted *copy* of the `DataFrame`. To modify `people` directly, we can set the `inplace` argument to `True`. Also, we can sort the columns instead of the rows by setting `axis=1`:


```python
people.sort_index(axis=1, inplace=True)
people
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>body_mass_index</th>
      <th>height</th>
      <th>hobby</th>
      <th>over 30</th>
      <th>overweight</th>
      <th>pets</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>33</td>
      <td>22.985398</td>
      <td>172</td>
      <td>Biking</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
      <td>68</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>34</td>
      <td>25.335002</td>
      <td>181</td>
      <td>Dancing</td>
      <td>True</td>
      <td>False</td>
      <td>0.0</td>
      <td>83</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>26</td>
      <td>32.724617</td>
      <td>185</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>5.0</td>
      <td>112</td>
    </tr>
  </tbody>
</table>
</div>



To sort the `DataFrame` by the values instead of the labels, we can use `sort_values` and specify the column to sort by:


```python
people.sort_values(by="age", inplace=True)
people
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>body_mass_index</th>
      <th>height</th>
      <th>hobby</th>
      <th>over 30</th>
      <th>overweight</th>
      <th>pets</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>charles</th>
      <td>26</td>
      <td>32.724617</td>
      <td>185</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>5.0</td>
      <td>112</td>
    </tr>
    <tr>
      <th>alice</th>
      <td>33</td>
      <td>22.985398</td>
      <td>172</td>
      <td>Biking</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
      <td>68</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>34</td>
      <td>25.335002</td>
      <td>181</td>
      <td>Dancing</td>
      <td>True</td>
      <td>False</td>
      <td>0.0</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>



## Plotting a `DataFrame`
Just like for `Series`, pandas makes it easy to draw nice graphs based on a `DataFrame`.

For example, it is trivial to create a line plot from a `DataFrame`'s data by calling its `plot` method:


```python
people.sort_values(by="body_mass_index", inplace=True)
people.plot(kind="line", x="body_mass_index", y=["height", "weight"])
plt.show()
```


    
![png](PANDAS_183_0.png)
    


You can pass extra arguments supported by matplotlib's functions. For example, we can create scatterplot and pass it a list of sizes using the `s` argument of matplotlib's `scatter()` function:


```python
people.plot(kind="scatter", x="height", y="weight", s=[40, 120, 200])
plt.show()
```


    
![png](PANDAS_185_0.png)
    


Again, there are way too many options to list here: the best option is to scroll through the [Visualization](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html) page in pandas' documentation, find the plot you are interested in and look at the example code.

## Operations on `DataFrame`s
Although `DataFrame`s do not try to mimic NumPy arrays, there are a few similarities. Let's create a `DataFrame` to demonstrate this:


```python
grades_array = np.array([[8, 8, 9], [10, 9, 9], [4, 8, 2], [9, 10, 10]])
grades = pd.DataFrame(grades_array, columns=["sep", "oct", "nov"], index=["alice", "bob", "charles", "darwin"])
grades
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sep</th>
      <th>oct</th>
      <th>nov</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>8</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>10</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>4</td>
      <td>8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>9</td>
      <td>10</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



You can apply NumPy mathematical functions on a `DataFrame`: the function is applied to all values:


```python
np.sqrt(grades)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sep</th>
      <th>oct</th>
      <th>nov</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>2.828427</td>
      <td>2.828427</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>3.162278</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>2.000000</td>
      <td>2.828427</td>
      <td>1.414214</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>3.000000</td>
      <td>3.162278</td>
      <td>3.162278</td>
    </tr>
  </tbody>
</table>
</div>



Similarly, adding a single value to a `DataFrame` will add that value to all elements in the `DataFrame`. This is called *broadcasting*:


```python
grades + 1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sep</th>
      <th>oct</th>
      <th>nov</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>9</td>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>11</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>5</td>
      <td>9</td>
      <td>3</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>10</td>
      <td>11</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



Of course, the same is true for all other binary operations, including arithmetic (`*`,`/`,`**`...) and conditional (`>`, `==`...) operations:


```python
grades >= 5
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sep</th>
      <th>oct</th>
      <th>nov</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



Aggregation operations, such as computing the `max`, the `sum` or the `mean` of a `DataFrame`, apply to each column, and you get back a `Series` object:


```python
grades.mean()
```




    sep    7.75
    oct    8.75
    nov    7.50
    dtype: float64



The `all` method is also an aggregation operation: it checks whether all values are `True` or not. Let's see during which months all students got a grade greater than `5`:


```python
(grades > 5).all()
```




    sep    False
    oct     True
    nov    False
    dtype: bool



Most of these functions take an optional `axis` parameter which lets you specify along which axis of the `DataFrame` you want the operation executed. The default is `axis=0`, meaning that the operation is executed vertically (on each column). You can set `axis=1` to execute the operation horizontally (on each row). For example, let's find out which students had all grades greater than `5`:


```python
(grades > 5).all(axis=1)
```




    alice       True
    bob         True
    charles    False
    darwin      True
    dtype: bool



The `any` method returns `True` if any value is True. Let's see who got at least one grade 10:


```python
(grades == 10).any(axis=1)
```




    alice      False
    bob         True
    charles    False
    darwin      True
    dtype: bool



If you add a `Series` object to a `DataFrame` (or execute any other binary operation), pandas attempts to broadcast the operation to all *rows* in the `DataFrame`. This only works if the `Series` has the same size as the `DataFrame`s rows. For example, let's subtract the `mean` of the `DataFrame` (a `Series` object) from the `DataFrame`:


```python
grades - grades.mean()  # equivalent to: grades - [7.75, 8.75, 7.50]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sep</th>
      <th>oct</th>
      <th>nov</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>0.25</td>
      <td>-0.75</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>2.25</td>
      <td>0.25</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>-3.75</td>
      <td>-0.75</td>
      <td>-5.5</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>1.25</td>
      <td>1.25</td>
      <td>2.5</td>
    </tr>
  </tbody>
</table>
</div>



We subtracted `7.75` from all September grades, `8.75` from October grades and `7.50` from November grades. It is equivalent to subtracting this `DataFrame`:


```python
pd.DataFrame([[7.75, 8.75, 7.50]]*4, index=grades.index, columns=grades.columns)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sep</th>
      <th>oct</th>
      <th>nov</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>7.75</td>
      <td>8.75</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>7.75</td>
      <td>8.75</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>7.75</td>
      <td>8.75</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>7.75</td>
      <td>8.75</td>
      <td>7.5</td>
    </tr>
  </tbody>
</table>
</div>



If you want to subtract the global mean from every grade, here is one way to do it:


```python
grades - grades.values.mean() # subtracts the global mean (8.00) from all grades
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sep</th>
      <th>oct</th>
      <th>nov</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>-4.0</td>
      <td>0.0</td>
      <td>-6.0</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



## Automatic alignment
Similar to `Series`, when operating on multiple `DataFrame`s, pandas automatically aligns them by row index label, but also by column names. Let's create a `DataFrame` with bonus points for each person from October to December:


```python
bonus_array = np.array([[0, np.nan, 2], [np.nan, 1, 0], [0, 1, 0], [3, 3, 0]])
bonus_points = pd.DataFrame(bonus_array, columns=["oct", "nov", "dec"], index=["bob", "colin", "darwin", "charles"])
bonus_points
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>oct</th>
      <th>nov</th>
      <th>dec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bob</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>colin</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
grades + bonus_points
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dec</th>
      <th>nov</th>
      <th>oct</th>
      <th>sep</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>NaN</td>
      <td>5.0</td>
      <td>11.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>colin</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>NaN</td>
      <td>11.0</td>
      <td>10.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Looks like the addition worked in some cases but way too many elements are now empty. That's because when aligning the `DataFrame`s, some columns and rows were only present on one side, and thus they were considered missing on the other side (`NaN`). Then adding `NaN` to a number results in `NaN`, hence the result.

## Handling missing data
Dealing with missing data is a frequent task when working with real life data. Pandas offers a few tools to handle missing data.

Let's try to fix the problem above. For example, we can decide that missing data should result in a zero, instead of `NaN`. We can replace all `NaN` values by any value using the `fillna()` method:


```python
(grades + bonus_points).fillna(0)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dec</th>
      <th>nov</th>
      <th>oct</th>
      <th>sep</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>0.0</td>
      <td>5.0</td>
      <td>11.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>colin</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>0.0</td>
      <td>11.0</td>
      <td>10.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



It's a bit unfair that we're setting grades to zero in September, though. Perhaps we should decide that missing grades are missing grades, but missing bonus points should be replaced by zeros:


```python
fixed_bonus_points = bonus_points.fillna(0)
fixed_bonus_points.insert(0, "sep", 0)
fixed_bonus_points.loc["alice"] = 0
grades + fixed_bonus_points
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dec</th>
      <th>nov</th>
      <th>oct</th>
      <th>sep</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>NaN</td>
      <td>9.0</td>
      <td>8.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>NaN</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>NaN</td>
      <td>5.0</td>
      <td>11.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>colin</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>NaN</td>
      <td>11.0</td>
      <td>10.0</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>



That's much better: although we made up some data, we have not been too unfair.

Another way to handle missing data is to interpolate. Let's look at the `bonus_points` `DataFrame` again:


```python
bonus_points
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>oct</th>
      <th>nov</th>
      <th>dec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bob</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>colin</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Now let's call the `interpolate` method. By default, it interpolates vertically (`axis=0`), so let's tell it to interpolate horizontally (`axis=1`).


```python
bonus_points.interpolate(axis=1)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>oct</th>
      <th>nov</th>
      <th>dec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bob</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>colin</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Bob had 0 bonus points in October, and 2 in December. When we interpolate for November, we get the mean: 1 bonus point. Colin had 1 bonus point in November, but we do not know how many bonus points he had in September, so we cannot interpolate, this is why there is still a missing value in October after interpolation. To fix this, we can set the September bonus points to 0 before interpolation.


```python
better_bonus_points = bonus_points.copy()
better_bonus_points.insert(0, "sep", 0)
better_bonus_points.loc["alice"] = 0
better_bonus_points = better_bonus_points.interpolate(axis=1)
better_bonus_points
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sep</th>
      <th>oct</th>
      <th>nov</th>
      <th>dec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bob</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>colin</th>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>alice</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Great, now we have reasonable bonus points everywhere. Let's find out the final grades:


```python
grades + better_bonus_points
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dec</th>
      <th>nov</th>
      <th>oct</th>
      <th>sep</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>NaN</td>
      <td>9.0</td>
      <td>8.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>NaN</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>NaN</td>
      <td>5.0</td>
      <td>11.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>colin</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>NaN</td>
      <td>11.0</td>
      <td>10.0</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>



It is slightly annoying that the September column ends up on the right. This is because the `DataFrame`s we are adding do not have the exact same columns (the `grades` `DataFrame` is missing the `"dec"` column), so to make things predictable, pandas orders the final columns alphabetically. To fix this, we can simply add the missing column before adding:


```python
grades["dec"] = np.nan
final_grades = grades + better_bonus_points
final_grades
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sep</th>
      <th>oct</th>
      <th>nov</th>
      <th>dec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>8.0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>10.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>4.0</td>
      <td>11.0</td>
      <td>5.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>colin</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>9.0</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



There's not much we can do about December and Colin: it's bad enough that we are making up bonus points, but we can't reasonably make up grades (well, I guess some teachers probably do). So let's call the `dropna()` method to get rid of rows that are full of `NaN`s:


```python
final_grades_clean = final_grades.dropna(how="all")
final_grades_clean
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sep</th>
      <th>oct</th>
      <th>nov</th>
      <th>dec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>8.0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>10.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>4.0</td>
      <td>11.0</td>
      <td>5.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>9.0</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Now let's remove columns that are full of `NaN`s by setting the `axis` argument to `1`:


```python
final_grades_clean = final_grades_clean.dropna(axis=1, how="all")
final_grades_clean
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sep</th>
      <th>oct</th>
      <th>nov</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>8.0</td>
      <td>8.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>10.0</td>
      <td>9.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>4.0</td>
      <td>11.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>9.0</td>
      <td>10.0</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>



## Aggregating with `groupby`
Similar to the SQL language, pandas allows grouping your data into groups to run calculations over each group.

First, let's add some extra data about each person so we can group them, and let's go back to the `final_grades` `DataFrame` so we can see how `NaN` values are handled:


```python
final_grades["hobby"] = ["Biking", "Dancing", np.nan, "Dancing", "Biking"]
final_grades
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sep</th>
      <th>oct</th>
      <th>nov</th>
      <th>dec</th>
      <th>hobby</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>8.0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>NaN</td>
      <td>Biking</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>10.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>Dancing</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>4.0</td>
      <td>11.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>colin</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Dancing</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>9.0</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>Biking</td>
    </tr>
  </tbody>
</table>
</div>



Now let's group data in this `DataFrame` by hobby:


```python
grouped_grades = final_grades.groupby("hobby")
grouped_grades
```




    <pandas.core.groupby.DataFrameGroupBy object at 0x10b680e10>



We are ready to compute the average grade per hobby:


```python
grouped_grades.mean()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sep</th>
      <th>oct</th>
      <th>nov</th>
      <th>dec</th>
    </tr>
    <tr>
      <th>hobby</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Biking</th>
      <td>8.5</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Dancing</th>
      <td>10.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



That was easy! Note that the `NaN` values have simply been skipped when computing the means.

## Pivot tables
Pandas supports spreadsheet-like [pivot tables](https://en.wikipedia.org/wiki/Pivot_table) that allow quick data summarization. To illustrate this, let's create a simple `DataFrame`:


```python
bonus_points
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>oct</th>
      <th>nov</th>
      <th>dec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bob</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>colin</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
more_grades = final_grades_clean.stack().reset_index()
more_grades.columns = ["name", "month", "grade"]
more_grades["bonus"] = [np.nan, np.nan, np.nan, 0, np.nan, 2, 3, 3, 0, 0, 1, 0]
more_grades
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>month</th>
      <th>grade</th>
      <th>bonus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>alice</td>
      <td>sep</td>
      <td>8.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>alice</td>
      <td>oct</td>
      <td>8.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>alice</td>
      <td>nov</td>
      <td>9.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>sep</td>
      <td>10.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bob</td>
      <td>oct</td>
      <td>9.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>bob</td>
      <td>nov</td>
      <td>10.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>charles</td>
      <td>sep</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>charles</td>
      <td>oct</td>
      <td>11.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>charles</td>
      <td>nov</td>
      <td>5.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>darwin</td>
      <td>sep</td>
      <td>9.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>darwin</td>
      <td>oct</td>
      <td>10.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>darwin</td>
      <td>nov</td>
      <td>11.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Now we can call the `pd.pivot_table()` function for this `DataFrame`, asking to group by the `name` column. By default, `pivot_table()` computes the mean of each numeric column:


```python
pd.pivot_table(more_grades, index="name")
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bonus</th>
      <th>grade</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>NaN</td>
      <td>8.333333</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>1.000000</td>
      <td>9.666667</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>2.000000</td>
      <td>6.666667</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>0.333333</td>
      <td>10.000000</td>
    </tr>
  </tbody>
</table>
</div>



We can change the aggregation function by setting the `aggfunc` argument, and we can also specify the list of columns whose values will be aggregated:


```python
pd.pivot_table(more_grades, index="name", values=["grade", "bonus"], aggfunc=np.max)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bonus</th>
      <th>grade</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>NaN</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>2.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>3.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>1.0</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>



We can also specify the `columns` to aggregate over horizontally, and request the grand totals for each row and column by setting `margins=True`:


```python
pd.pivot_table(more_grades, index="name", values="grade", columns="month", margins=True)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>month</th>
      <th>nov</th>
      <th>oct</th>
      <th>sep</th>
      <th>All</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>9.00</td>
      <td>8.0</td>
      <td>8.00</td>
      <td>8.333333</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>10.00</td>
      <td>9.0</td>
      <td>10.00</td>
      <td>9.666667</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>5.00</td>
      <td>11.0</td>
      <td>4.00</td>
      <td>6.666667</td>
    </tr>
    <tr>
      <th>darwin</th>
      <td>11.00</td>
      <td>10.0</td>
      <td>9.00</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>All</th>
      <td>8.75</td>
      <td>9.5</td>
      <td>7.75</td>
      <td>8.666667</td>
    </tr>
  </tbody>
</table>
</div>



Finally, we can specify multiple index or column names, and pandas will create multi-level indices:


```python
pd.pivot_table(more_grades, index=("name", "month"), margins=True)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>bonus</th>
      <th>grade</th>
    </tr>
    <tr>
      <th>name</th>
      <th>month</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">alice</th>
      <th>nov</th>
      <td>NaN</td>
      <td>9.00</td>
    </tr>
    <tr>
      <th>oct</th>
      <td>NaN</td>
      <td>8.00</td>
    </tr>
    <tr>
      <th>sep</th>
      <td>NaN</td>
      <td>8.00</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">bob</th>
      <th>nov</th>
      <td>2.000</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>oct</th>
      <td>NaN</td>
      <td>9.00</td>
    </tr>
    <tr>
      <th>sep</th>
      <td>0.000</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">charles</th>
      <th>nov</th>
      <td>0.000</td>
      <td>5.00</td>
    </tr>
    <tr>
      <th>oct</th>
      <td>3.000</td>
      <td>11.00</td>
    </tr>
    <tr>
      <th>sep</th>
      <td>3.000</td>
      <td>4.00</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">darwin</th>
      <th>nov</th>
      <td>0.000</td>
      <td>11.00</td>
    </tr>
    <tr>
      <th>oct</th>
      <td>1.000</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>sep</th>
      <td>0.000</td>
      <td>9.00</td>
    </tr>
    <tr>
      <th>All</th>
      <th></th>
      <td>1.125</td>
      <td>8.75</td>
    </tr>
  </tbody>
</table>
</div>



## Overview functions
When dealing with large `DataFrames`, it is useful to get a quick overview of its content. Pandas offers a few functions for this. First, let's create a large `DataFrame` with a mix of numeric values, missing values and text values. Notice how Jupyter displays only the corners of the `DataFrame`:


```python
much_data = np.fromfunction(lambda x,y: (x+y*y)%17*11, (10000, 26))
large_df = pd.DataFrame(much_data, columns=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
large_df[large_df % 16 == 0] = np.nan
large_df.insert(3, "some_text", "Blabla")
large_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>some_text</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>G</th>
      <th>H</th>
      <th>I</th>
      <th>...</th>
      <th>Q</th>
      <th>R</th>
      <th>S</th>
      <th>T</th>
      <th>U</th>
      <th>V</th>
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>11.0</td>
      <td>44.0</td>
      <td>Blabla</td>
      <td>99.0</td>
      <td>NaN</td>
      <td>88.0</td>
      <td>22.0</td>
      <td>165.0</td>
      <td>143.0</td>
      <td>...</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>44.0</td>
      <td>99.0</td>
      <td>NaN</td>
      <td>88.0</td>
      <td>22.0</td>
      <td>165.0</td>
      <td>143.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11.0</td>
      <td>22.0</td>
      <td>55.0</td>
      <td>Blabla</td>
      <td>110.0</td>
      <td>NaN</td>
      <td>99.0</td>
      <td>33.0</td>
      <td>NaN</td>
      <td>154.0</td>
      <td>...</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>22.0</td>
      <td>55.0</td>
      <td>110.0</td>
      <td>NaN</td>
      <td>99.0</td>
      <td>33.0</td>
      <td>NaN</td>
      <td>154.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22.0</td>
      <td>33.0</td>
      <td>66.0</td>
      <td>Blabla</td>
      <td>121.0</td>
      <td>11.0</td>
      <td>110.0</td>
      <td>44.0</td>
      <td>NaN</td>
      <td>165.0</td>
      <td>...</td>
      <td>33.0</td>
      <td>22.0</td>
      <td>33.0</td>
      <td>66.0</td>
      <td>121.0</td>
      <td>11.0</td>
      <td>110.0</td>
      <td>44.0</td>
      <td>NaN</td>
      <td>165.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33.0</td>
      <td>44.0</td>
      <td>77.0</td>
      <td>Blabla</td>
      <td>132.0</td>
      <td>22.0</td>
      <td>121.0</td>
      <td>55.0</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>44.0</td>
      <td>33.0</td>
      <td>44.0</td>
      <td>77.0</td>
      <td>132.0</td>
      <td>22.0</td>
      <td>121.0</td>
      <td>55.0</td>
      <td>11.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>44.0</td>
      <td>55.0</td>
      <td>88.0</td>
      <td>Blabla</td>
      <td>143.0</td>
      <td>33.0</td>
      <td>132.0</td>
      <td>66.0</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>55.0</td>
      <td>44.0</td>
      <td>55.0</td>
      <td>88.0</td>
      <td>143.0</td>
      <td>33.0</td>
      <td>132.0</td>
      <td>66.0</td>
      <td>22.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>55.0</td>
      <td>66.0</td>
      <td>99.0</td>
      <td>Blabla</td>
      <td>154.0</td>
      <td>44.0</td>
      <td>143.0</td>
      <td>77.0</td>
      <td>33.0</td>
      <td>11.0</td>
      <td>...</td>
      <td>66.0</td>
      <td>55.0</td>
      <td>66.0</td>
      <td>99.0</td>
      <td>154.0</td>
      <td>44.0</td>
      <td>143.0</td>
      <td>77.0</td>
      <td>33.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>66.0</td>
      <td>77.0</td>
      <td>110.0</td>
      <td>Blabla</td>
      <td>165.0</td>
      <td>55.0</td>
      <td>154.0</td>
      <td>88.0</td>
      <td>44.0</td>
      <td>22.0</td>
      <td>...</td>
      <td>77.0</td>
      <td>66.0</td>
      <td>77.0</td>
      <td>110.0</td>
      <td>165.0</td>
      <td>55.0</td>
      <td>154.0</td>
      <td>88.0</td>
      <td>44.0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>77.0</td>
      <td>88.0</td>
      <td>121.0</td>
      <td>Blabla</td>
      <td>NaN</td>
      <td>66.0</td>
      <td>165.0</td>
      <td>99.0</td>
      <td>55.0</td>
      <td>33.0</td>
      <td>...</td>
      <td>88.0</td>
      <td>77.0</td>
      <td>88.0</td>
      <td>121.0</td>
      <td>NaN</td>
      <td>66.0</td>
      <td>165.0</td>
      <td>99.0</td>
      <td>55.0</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>88.0</td>
      <td>99.0</td>
      <td>132.0</td>
      <td>Blabla</td>
      <td>NaN</td>
      <td>77.0</td>
      <td>NaN</td>
      <td>110.0</td>
      <td>66.0</td>
      <td>44.0</td>
      <td>...</td>
      <td>99.0</td>
      <td>88.0</td>
      <td>99.0</td>
      <td>132.0</td>
      <td>NaN</td>
      <td>77.0</td>
      <td>NaN</td>
      <td>110.0</td>
      <td>66.0</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>99.0</td>
      <td>110.0</td>
      <td>143.0</td>
      <td>Blabla</td>
      <td>11.0</td>
      <td>88.0</td>
      <td>NaN</td>
      <td>121.0</td>
      <td>77.0</td>
      <td>55.0</td>
      <td>...</td>
      <td>110.0</td>
      <td>99.0</td>
      <td>110.0</td>
      <td>143.0</td>
      <td>11.0</td>
      <td>88.0</td>
      <td>NaN</td>
      <td>121.0</td>
      <td>77.0</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>110.0</td>
      <td>121.0</td>
      <td>154.0</td>
      <td>Blabla</td>
      <td>22.0</td>
      <td>99.0</td>
      <td>11.0</td>
      <td>132.0</td>
      <td>88.0</td>
      <td>66.0</td>
      <td>...</td>
      <td>121.0</td>
      <td>110.0</td>
      <td>121.0</td>
      <td>154.0</td>
      <td>22.0</td>
      <td>99.0</td>
      <td>11.0</td>
      <td>132.0</td>
      <td>88.0</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>121.0</td>
      <td>132.0</td>
      <td>165.0</td>
      <td>Blabla</td>
      <td>33.0</td>
      <td>110.0</td>
      <td>22.0</td>
      <td>143.0</td>
      <td>99.0</td>
      <td>77.0</td>
      <td>...</td>
      <td>132.0</td>
      <td>121.0</td>
      <td>132.0</td>
      <td>165.0</td>
      <td>33.0</td>
      <td>110.0</td>
      <td>22.0</td>
      <td>143.0</td>
      <td>99.0</td>
      <td>77.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>132.0</td>
      <td>143.0</td>
      <td>NaN</td>
      <td>Blabla</td>
      <td>44.0</td>
      <td>121.0</td>
      <td>33.0</td>
      <td>154.0</td>
      <td>110.0</td>
      <td>88.0</td>
      <td>...</td>
      <td>143.0</td>
      <td>132.0</td>
      <td>143.0</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>121.0</td>
      <td>33.0</td>
      <td>154.0</td>
      <td>110.0</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>143.0</td>
      <td>154.0</td>
      <td>NaN</td>
      <td>Blabla</td>
      <td>55.0</td>
      <td>132.0</td>
      <td>44.0</td>
      <td>165.0</td>
      <td>121.0</td>
      <td>99.0</td>
      <td>...</td>
      <td>154.0</td>
      <td>143.0</td>
      <td>154.0</td>
      <td>NaN</td>
      <td>55.0</td>
      <td>132.0</td>
      <td>44.0</td>
      <td>165.0</td>
      <td>121.0</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>154.0</td>
      <td>165.0</td>
      <td>11.0</td>
      <td>Blabla</td>
      <td>66.0</td>
      <td>143.0</td>
      <td>55.0</td>
      <td>NaN</td>
      <td>132.0</td>
      <td>110.0</td>
      <td>...</td>
      <td>165.0</td>
      <td>154.0</td>
      <td>165.0</td>
      <td>11.0</td>
      <td>66.0</td>
      <td>143.0</td>
      <td>55.0</td>
      <td>NaN</td>
      <td>132.0</td>
      <td>110.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>165.0</td>
      <td>NaN</td>
      <td>22.0</td>
      <td>Blabla</td>
      <td>77.0</td>
      <td>154.0</td>
      <td>66.0</td>
      <td>NaN</td>
      <td>143.0</td>
      <td>121.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>165.0</td>
      <td>NaN</td>
      <td>22.0</td>
      <td>77.0</td>
      <td>154.0</td>
      <td>66.0</td>
      <td>NaN</td>
      <td>143.0</td>
      <td>121.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>33.0</td>
      <td>Blabla</td>
      <td>88.0</td>
      <td>165.0</td>
      <td>77.0</td>
      <td>11.0</td>
      <td>154.0</td>
      <td>132.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>33.0</td>
      <td>88.0</td>
      <td>165.0</td>
      <td>77.0</td>
      <td>11.0</td>
      <td>154.0</td>
      <td>132.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>NaN</td>
      <td>11.0</td>
      <td>44.0</td>
      <td>Blabla</td>
      <td>99.0</td>
      <td>NaN</td>
      <td>88.0</td>
      <td>22.0</td>
      <td>165.0</td>
      <td>143.0</td>
      <td>...</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>44.0</td>
      <td>99.0</td>
      <td>NaN</td>
      <td>88.0</td>
      <td>22.0</td>
      <td>165.0</td>
      <td>143.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>11.0</td>
      <td>22.0</td>
      <td>55.0</td>
      <td>Blabla</td>
      <td>110.0</td>
      <td>NaN</td>
      <td>99.0</td>
      <td>33.0</td>
      <td>NaN</td>
      <td>154.0</td>
      <td>...</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>22.0</td>
      <td>55.0</td>
      <td>110.0</td>
      <td>NaN</td>
      <td>99.0</td>
      <td>33.0</td>
      <td>NaN</td>
      <td>154.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>22.0</td>
      <td>33.0</td>
      <td>66.0</td>
      <td>Blabla</td>
      <td>121.0</td>
      <td>11.0</td>
      <td>110.0</td>
      <td>44.0</td>
      <td>NaN</td>
      <td>165.0</td>
      <td>...</td>
      <td>33.0</td>
      <td>22.0</td>
      <td>33.0</td>
      <td>66.0</td>
      <td>121.0</td>
      <td>11.0</td>
      <td>110.0</td>
      <td>44.0</td>
      <td>NaN</td>
      <td>165.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>33.0</td>
      <td>44.0</td>
      <td>77.0</td>
      <td>Blabla</td>
      <td>132.0</td>
      <td>22.0</td>
      <td>121.0</td>
      <td>55.0</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>44.0</td>
      <td>33.0</td>
      <td>44.0</td>
      <td>77.0</td>
      <td>132.0</td>
      <td>22.0</td>
      <td>121.0</td>
      <td>55.0</td>
      <td>11.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>44.0</td>
      <td>55.0</td>
      <td>88.0</td>
      <td>Blabla</td>
      <td>143.0</td>
      <td>33.0</td>
      <td>132.0</td>
      <td>66.0</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>55.0</td>
      <td>44.0</td>
      <td>55.0</td>
      <td>88.0</td>
      <td>143.0</td>
      <td>33.0</td>
      <td>132.0</td>
      <td>66.0</td>
      <td>22.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>55.0</td>
      <td>66.0</td>
      <td>99.0</td>
      <td>Blabla</td>
      <td>154.0</td>
      <td>44.0</td>
      <td>143.0</td>
      <td>77.0</td>
      <td>33.0</td>
      <td>11.0</td>
      <td>...</td>
      <td>66.0</td>
      <td>55.0</td>
      <td>66.0</td>
      <td>99.0</td>
      <td>154.0</td>
      <td>44.0</td>
      <td>143.0</td>
      <td>77.0</td>
      <td>33.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>66.0</td>
      <td>77.0</td>
      <td>110.0</td>
      <td>Blabla</td>
      <td>165.0</td>
      <td>55.0</td>
      <td>154.0</td>
      <td>88.0</td>
      <td>44.0</td>
      <td>22.0</td>
      <td>...</td>
      <td>77.0</td>
      <td>66.0</td>
      <td>77.0</td>
      <td>110.0</td>
      <td>165.0</td>
      <td>55.0</td>
      <td>154.0</td>
      <td>88.0</td>
      <td>44.0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>77.0</td>
      <td>88.0</td>
      <td>121.0</td>
      <td>Blabla</td>
      <td>NaN</td>
      <td>66.0</td>
      <td>165.0</td>
      <td>99.0</td>
      <td>55.0</td>
      <td>33.0</td>
      <td>...</td>
      <td>88.0</td>
      <td>77.0</td>
      <td>88.0</td>
      <td>121.0</td>
      <td>NaN</td>
      <td>66.0</td>
      <td>165.0</td>
      <td>99.0</td>
      <td>55.0</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>88.0</td>
      <td>99.0</td>
      <td>132.0</td>
      <td>Blabla</td>
      <td>NaN</td>
      <td>77.0</td>
      <td>NaN</td>
      <td>110.0</td>
      <td>66.0</td>
      <td>44.0</td>
      <td>...</td>
      <td>99.0</td>
      <td>88.0</td>
      <td>99.0</td>
      <td>132.0</td>
      <td>NaN</td>
      <td>77.0</td>
      <td>NaN</td>
      <td>110.0</td>
      <td>66.0</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>99.0</td>
      <td>110.0</td>
      <td>143.0</td>
      <td>Blabla</td>
      <td>11.0</td>
      <td>88.0</td>
      <td>NaN</td>
      <td>121.0</td>
      <td>77.0</td>
      <td>55.0</td>
      <td>...</td>
      <td>110.0</td>
      <td>99.0</td>
      <td>110.0</td>
      <td>143.0</td>
      <td>11.0</td>
      <td>88.0</td>
      <td>NaN</td>
      <td>121.0</td>
      <td>77.0</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>110.0</td>
      <td>121.0</td>
      <td>154.0</td>
      <td>Blabla</td>
      <td>22.0</td>
      <td>99.0</td>
      <td>11.0</td>
      <td>132.0</td>
      <td>88.0</td>
      <td>66.0</td>
      <td>...</td>
      <td>121.0</td>
      <td>110.0</td>
      <td>121.0</td>
      <td>154.0</td>
      <td>22.0</td>
      <td>99.0</td>
      <td>11.0</td>
      <td>132.0</td>
      <td>88.0</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>121.0</td>
      <td>132.0</td>
      <td>165.0</td>
      <td>Blabla</td>
      <td>33.0</td>
      <td>110.0</td>
      <td>22.0</td>
      <td>143.0</td>
      <td>99.0</td>
      <td>77.0</td>
      <td>...</td>
      <td>132.0</td>
      <td>121.0</td>
      <td>132.0</td>
      <td>165.0</td>
      <td>33.0</td>
      <td>110.0</td>
      <td>22.0</td>
      <td>143.0</td>
      <td>99.0</td>
      <td>77.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>132.0</td>
      <td>143.0</td>
      <td>NaN</td>
      <td>Blabla</td>
      <td>44.0</td>
      <td>121.0</td>
      <td>33.0</td>
      <td>154.0</td>
      <td>110.0</td>
      <td>88.0</td>
      <td>...</td>
      <td>143.0</td>
      <td>132.0</td>
      <td>143.0</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>121.0</td>
      <td>33.0</td>
      <td>154.0</td>
      <td>110.0</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9970</th>
      <td>88.0</td>
      <td>99.0</td>
      <td>132.0</td>
      <td>Blabla</td>
      <td>NaN</td>
      <td>77.0</td>
      <td>NaN</td>
      <td>110.0</td>
      <td>66.0</td>
      <td>44.0</td>
      <td>...</td>
      <td>99.0</td>
      <td>88.0</td>
      <td>99.0</td>
      <td>132.0</td>
      <td>NaN</td>
      <td>77.0</td>
      <td>NaN</td>
      <td>110.0</td>
      <td>66.0</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>9971</th>
      <td>99.0</td>
      <td>110.0</td>
      <td>143.0</td>
      <td>Blabla</td>
      <td>11.0</td>
      <td>88.0</td>
      <td>NaN</td>
      <td>121.0</td>
      <td>77.0</td>
      <td>55.0</td>
      <td>...</td>
      <td>110.0</td>
      <td>99.0</td>
      <td>110.0</td>
      <td>143.0</td>
      <td>11.0</td>
      <td>88.0</td>
      <td>NaN</td>
      <td>121.0</td>
      <td>77.0</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>9972</th>
      <td>110.0</td>
      <td>121.0</td>
      <td>154.0</td>
      <td>Blabla</td>
      <td>22.0</td>
      <td>99.0</td>
      <td>11.0</td>
      <td>132.0</td>
      <td>88.0</td>
      <td>66.0</td>
      <td>...</td>
      <td>121.0</td>
      <td>110.0</td>
      <td>121.0</td>
      <td>154.0</td>
      <td>22.0</td>
      <td>99.0</td>
      <td>11.0</td>
      <td>132.0</td>
      <td>88.0</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>9973</th>
      <td>121.0</td>
      <td>132.0</td>
      <td>165.0</td>
      <td>Blabla</td>
      <td>33.0</td>
      <td>110.0</td>
      <td>22.0</td>
      <td>143.0</td>
      <td>99.0</td>
      <td>77.0</td>
      <td>...</td>
      <td>132.0</td>
      <td>121.0</td>
      <td>132.0</td>
      <td>165.0</td>
      <td>33.0</td>
      <td>110.0</td>
      <td>22.0</td>
      <td>143.0</td>
      <td>99.0</td>
      <td>77.0</td>
    </tr>
    <tr>
      <th>9974</th>
      <td>132.0</td>
      <td>143.0</td>
      <td>NaN</td>
      <td>Blabla</td>
      <td>44.0</td>
      <td>121.0</td>
      <td>33.0</td>
      <td>154.0</td>
      <td>110.0</td>
      <td>88.0</td>
      <td>...</td>
      <td>143.0</td>
      <td>132.0</td>
      <td>143.0</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>121.0</td>
      <td>33.0</td>
      <td>154.0</td>
      <td>110.0</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>9975</th>
      <td>143.0</td>
      <td>154.0</td>
      <td>NaN</td>
      <td>Blabla</td>
      <td>55.0</td>
      <td>132.0</td>
      <td>44.0</td>
      <td>165.0</td>
      <td>121.0</td>
      <td>99.0</td>
      <td>...</td>
      <td>154.0</td>
      <td>143.0</td>
      <td>154.0</td>
      <td>NaN</td>
      <td>55.0</td>
      <td>132.0</td>
      <td>44.0</td>
      <td>165.0</td>
      <td>121.0</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>9976</th>
      <td>154.0</td>
      <td>165.0</td>
      <td>11.0</td>
      <td>Blabla</td>
      <td>66.0</td>
      <td>143.0</td>
      <td>55.0</td>
      <td>NaN</td>
      <td>132.0</td>
      <td>110.0</td>
      <td>...</td>
      <td>165.0</td>
      <td>154.0</td>
      <td>165.0</td>
      <td>11.0</td>
      <td>66.0</td>
      <td>143.0</td>
      <td>55.0</td>
      <td>NaN</td>
      <td>132.0</td>
      <td>110.0</td>
    </tr>
    <tr>
      <th>9977</th>
      <td>165.0</td>
      <td>NaN</td>
      <td>22.0</td>
      <td>Blabla</td>
      <td>77.0</td>
      <td>154.0</td>
      <td>66.0</td>
      <td>NaN</td>
      <td>143.0</td>
      <td>121.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>165.0</td>
      <td>NaN</td>
      <td>22.0</td>
      <td>77.0</td>
      <td>154.0</td>
      <td>66.0</td>
      <td>NaN</td>
      <td>143.0</td>
      <td>121.0</td>
    </tr>
    <tr>
      <th>9978</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>33.0</td>
      <td>Blabla</td>
      <td>88.0</td>
      <td>165.0</td>
      <td>77.0</td>
      <td>11.0</td>
      <td>154.0</td>
      <td>132.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>33.0</td>
      <td>88.0</td>
      <td>165.0</td>
      <td>77.0</td>
      <td>11.0</td>
      <td>154.0</td>
      <td>132.0</td>
    </tr>
    <tr>
      <th>9979</th>
      <td>NaN</td>
      <td>11.0</td>
      <td>44.0</td>
      <td>Blabla</td>
      <td>99.0</td>
      <td>NaN</td>
      <td>88.0</td>
      <td>22.0</td>
      <td>165.0</td>
      <td>143.0</td>
      <td>...</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>44.0</td>
      <td>99.0</td>
      <td>NaN</td>
      <td>88.0</td>
      <td>22.0</td>
      <td>165.0</td>
      <td>143.0</td>
    </tr>
    <tr>
      <th>9980</th>
      <td>11.0</td>
      <td>22.0</td>
      <td>55.0</td>
      <td>Blabla</td>
      <td>110.0</td>
      <td>NaN</td>
      <td>99.0</td>
      <td>33.0</td>
      <td>NaN</td>
      <td>154.0</td>
      <td>...</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>22.0</td>
      <td>55.0</td>
      <td>110.0</td>
      <td>NaN</td>
      <td>99.0</td>
      <td>33.0</td>
      <td>NaN</td>
      <td>154.0</td>
    </tr>
    <tr>
      <th>9981</th>
      <td>22.0</td>
      <td>33.0</td>
      <td>66.0</td>
      <td>Blabla</td>
      <td>121.0</td>
      <td>11.0</td>
      <td>110.0</td>
      <td>44.0</td>
      <td>NaN</td>
      <td>165.0</td>
      <td>...</td>
      <td>33.0</td>
      <td>22.0</td>
      <td>33.0</td>
      <td>66.0</td>
      <td>121.0</td>
      <td>11.0</td>
      <td>110.0</td>
      <td>44.0</td>
      <td>NaN</td>
      <td>165.0</td>
    </tr>
    <tr>
      <th>9982</th>
      <td>33.0</td>
      <td>44.0</td>
      <td>77.0</td>
      <td>Blabla</td>
      <td>132.0</td>
      <td>22.0</td>
      <td>121.0</td>
      <td>55.0</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>44.0</td>
      <td>33.0</td>
      <td>44.0</td>
      <td>77.0</td>
      <td>132.0</td>
      <td>22.0</td>
      <td>121.0</td>
      <td>55.0</td>
      <td>11.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9983</th>
      <td>44.0</td>
      <td>55.0</td>
      <td>88.0</td>
      <td>Blabla</td>
      <td>143.0</td>
      <td>33.0</td>
      <td>132.0</td>
      <td>66.0</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>55.0</td>
      <td>44.0</td>
      <td>55.0</td>
      <td>88.0</td>
      <td>143.0</td>
      <td>33.0</td>
      <td>132.0</td>
      <td>66.0</td>
      <td>22.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9984</th>
      <td>55.0</td>
      <td>66.0</td>
      <td>99.0</td>
      <td>Blabla</td>
      <td>154.0</td>
      <td>44.0</td>
      <td>143.0</td>
      <td>77.0</td>
      <td>33.0</td>
      <td>11.0</td>
      <td>...</td>
      <td>66.0</td>
      <td>55.0</td>
      <td>66.0</td>
      <td>99.0</td>
      <td>154.0</td>
      <td>44.0</td>
      <td>143.0</td>
      <td>77.0</td>
      <td>33.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>9985</th>
      <td>66.0</td>
      <td>77.0</td>
      <td>110.0</td>
      <td>Blabla</td>
      <td>165.0</td>
      <td>55.0</td>
      <td>154.0</td>
      <td>88.0</td>
      <td>44.0</td>
      <td>22.0</td>
      <td>...</td>
      <td>77.0</td>
      <td>66.0</td>
      <td>77.0</td>
      <td>110.0</td>
      <td>165.0</td>
      <td>55.0</td>
      <td>154.0</td>
      <td>88.0</td>
      <td>44.0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>9986</th>
      <td>77.0</td>
      <td>88.0</td>
      <td>121.0</td>
      <td>Blabla</td>
      <td>NaN</td>
      <td>66.0</td>
      <td>165.0</td>
      <td>99.0</td>
      <td>55.0</td>
      <td>33.0</td>
      <td>...</td>
      <td>88.0</td>
      <td>77.0</td>
      <td>88.0</td>
      <td>121.0</td>
      <td>NaN</td>
      <td>66.0</td>
      <td>165.0</td>
      <td>99.0</td>
      <td>55.0</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>9987</th>
      <td>88.0</td>
      <td>99.0</td>
      <td>132.0</td>
      <td>Blabla</td>
      <td>NaN</td>
      <td>77.0</td>
      <td>NaN</td>
      <td>110.0</td>
      <td>66.0</td>
      <td>44.0</td>
      <td>...</td>
      <td>99.0</td>
      <td>88.0</td>
      <td>99.0</td>
      <td>132.0</td>
      <td>NaN</td>
      <td>77.0</td>
      <td>NaN</td>
      <td>110.0</td>
      <td>66.0</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>9988</th>
      <td>99.0</td>
      <td>110.0</td>
      <td>143.0</td>
      <td>Blabla</td>
      <td>11.0</td>
      <td>88.0</td>
      <td>NaN</td>
      <td>121.0</td>
      <td>77.0</td>
      <td>55.0</td>
      <td>...</td>
      <td>110.0</td>
      <td>99.0</td>
      <td>110.0</td>
      <td>143.0</td>
      <td>11.0</td>
      <td>88.0</td>
      <td>NaN</td>
      <td>121.0</td>
      <td>77.0</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>9989</th>
      <td>110.0</td>
      <td>121.0</td>
      <td>154.0</td>
      <td>Blabla</td>
      <td>22.0</td>
      <td>99.0</td>
      <td>11.0</td>
      <td>132.0</td>
      <td>88.0</td>
      <td>66.0</td>
      <td>...</td>
      <td>121.0</td>
      <td>110.0</td>
      <td>121.0</td>
      <td>154.0</td>
      <td>22.0</td>
      <td>99.0</td>
      <td>11.0</td>
      <td>132.0</td>
      <td>88.0</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>9990</th>
      <td>121.0</td>
      <td>132.0</td>
      <td>165.0</td>
      <td>Blabla</td>
      <td>33.0</td>
      <td>110.0</td>
      <td>22.0</td>
      <td>143.0</td>
      <td>99.0</td>
      <td>77.0</td>
      <td>...</td>
      <td>132.0</td>
      <td>121.0</td>
      <td>132.0</td>
      <td>165.0</td>
      <td>33.0</td>
      <td>110.0</td>
      <td>22.0</td>
      <td>143.0</td>
      <td>99.0</td>
      <td>77.0</td>
    </tr>
    <tr>
      <th>9991</th>
      <td>132.0</td>
      <td>143.0</td>
      <td>NaN</td>
      <td>Blabla</td>
      <td>44.0</td>
      <td>121.0</td>
      <td>33.0</td>
      <td>154.0</td>
      <td>110.0</td>
      <td>88.0</td>
      <td>...</td>
      <td>143.0</td>
      <td>132.0</td>
      <td>143.0</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>121.0</td>
      <td>33.0</td>
      <td>154.0</td>
      <td>110.0</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>9992</th>
      <td>143.0</td>
      <td>154.0</td>
      <td>NaN</td>
      <td>Blabla</td>
      <td>55.0</td>
      <td>132.0</td>
      <td>44.0</td>
      <td>165.0</td>
      <td>121.0</td>
      <td>99.0</td>
      <td>...</td>
      <td>154.0</td>
      <td>143.0</td>
      <td>154.0</td>
      <td>NaN</td>
      <td>55.0</td>
      <td>132.0</td>
      <td>44.0</td>
      <td>165.0</td>
      <td>121.0</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>9993</th>
      <td>154.0</td>
      <td>165.0</td>
      <td>11.0</td>
      <td>Blabla</td>
      <td>66.0</td>
      <td>143.0</td>
      <td>55.0</td>
      <td>NaN</td>
      <td>132.0</td>
      <td>110.0</td>
      <td>...</td>
      <td>165.0</td>
      <td>154.0</td>
      <td>165.0</td>
      <td>11.0</td>
      <td>66.0</td>
      <td>143.0</td>
      <td>55.0</td>
      <td>NaN</td>
      <td>132.0</td>
      <td>110.0</td>
    </tr>
    <tr>
      <th>9994</th>
      <td>165.0</td>
      <td>NaN</td>
      <td>22.0</td>
      <td>Blabla</td>
      <td>77.0</td>
      <td>154.0</td>
      <td>66.0</td>
      <td>NaN</td>
      <td>143.0</td>
      <td>121.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>165.0</td>
      <td>NaN</td>
      <td>22.0</td>
      <td>77.0</td>
      <td>154.0</td>
      <td>66.0</td>
      <td>NaN</td>
      <td>143.0</td>
      <td>121.0</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>33.0</td>
      <td>Blabla</td>
      <td>88.0</td>
      <td>165.0</td>
      <td>77.0</td>
      <td>11.0</td>
      <td>154.0</td>
      <td>132.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>33.0</td>
      <td>88.0</td>
      <td>165.0</td>
      <td>77.0</td>
      <td>11.0</td>
      <td>154.0</td>
      <td>132.0</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>NaN</td>
      <td>11.0</td>
      <td>44.0</td>
      <td>Blabla</td>
      <td>99.0</td>
      <td>NaN</td>
      <td>88.0</td>
      <td>22.0</td>
      <td>165.0</td>
      <td>143.0</td>
      <td>...</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>44.0</td>
      <td>99.0</td>
      <td>NaN</td>
      <td>88.0</td>
      <td>22.0</td>
      <td>165.0</td>
      <td>143.0</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>11.0</td>
      <td>22.0</td>
      <td>55.0</td>
      <td>Blabla</td>
      <td>110.0</td>
      <td>NaN</td>
      <td>99.0</td>
      <td>33.0</td>
      <td>NaN</td>
      <td>154.0</td>
      <td>...</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>22.0</td>
      <td>55.0</td>
      <td>110.0</td>
      <td>NaN</td>
      <td>99.0</td>
      <td>33.0</td>
      <td>NaN</td>
      <td>154.0</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>22.0</td>
      <td>33.0</td>
      <td>66.0</td>
      <td>Blabla</td>
      <td>121.0</td>
      <td>11.0</td>
      <td>110.0</td>
      <td>44.0</td>
      <td>NaN</td>
      <td>165.0</td>
      <td>...</td>
      <td>33.0</td>
      <td>22.0</td>
      <td>33.0</td>
      <td>66.0</td>
      <td>121.0</td>
      <td>11.0</td>
      <td>110.0</td>
      <td>44.0</td>
      <td>NaN</td>
      <td>165.0</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>33.0</td>
      <td>44.0</td>
      <td>77.0</td>
      <td>Blabla</td>
      <td>132.0</td>
      <td>22.0</td>
      <td>121.0</td>
      <td>55.0</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>44.0</td>
      <td>33.0</td>
      <td>44.0</td>
      <td>77.0</td>
      <td>132.0</td>
      <td>22.0</td>
      <td>121.0</td>
      <td>55.0</td>
      <td>11.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 27 columns</p>
</div>



The `head()` method returns the top 5 rows:


```python
large_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>some_text</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>G</th>
      <th>H</th>
      <th>I</th>
      <th>...</th>
      <th>Q</th>
      <th>R</th>
      <th>S</th>
      <th>T</th>
      <th>U</th>
      <th>V</th>
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>11.0</td>
      <td>44.0</td>
      <td>Blabla</td>
      <td>99.0</td>
      <td>NaN</td>
      <td>88.0</td>
      <td>22.0</td>
      <td>165.0</td>
      <td>143.0</td>
      <td>...</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>44.0</td>
      <td>99.0</td>
      <td>NaN</td>
      <td>88.0</td>
      <td>22.0</td>
      <td>165.0</td>
      <td>143.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11.0</td>
      <td>22.0</td>
      <td>55.0</td>
      <td>Blabla</td>
      <td>110.0</td>
      <td>NaN</td>
      <td>99.0</td>
      <td>33.0</td>
      <td>NaN</td>
      <td>154.0</td>
      <td>...</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>22.0</td>
      <td>55.0</td>
      <td>110.0</td>
      <td>NaN</td>
      <td>99.0</td>
      <td>33.0</td>
      <td>NaN</td>
      <td>154.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22.0</td>
      <td>33.0</td>
      <td>66.0</td>
      <td>Blabla</td>
      <td>121.0</td>
      <td>11.0</td>
      <td>110.0</td>
      <td>44.0</td>
      <td>NaN</td>
      <td>165.0</td>
      <td>...</td>
      <td>33.0</td>
      <td>22.0</td>
      <td>33.0</td>
      <td>66.0</td>
      <td>121.0</td>
      <td>11.0</td>
      <td>110.0</td>
      <td>44.0</td>
      <td>NaN</td>
      <td>165.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33.0</td>
      <td>44.0</td>
      <td>77.0</td>
      <td>Blabla</td>
      <td>132.0</td>
      <td>22.0</td>
      <td>121.0</td>
      <td>55.0</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>44.0</td>
      <td>33.0</td>
      <td>44.0</td>
      <td>77.0</td>
      <td>132.0</td>
      <td>22.0</td>
      <td>121.0</td>
      <td>55.0</td>
      <td>11.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>44.0</td>
      <td>55.0</td>
      <td>88.0</td>
      <td>Blabla</td>
      <td>143.0</td>
      <td>33.0</td>
      <td>132.0</td>
      <td>66.0</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>55.0</td>
      <td>44.0</td>
      <td>55.0</td>
      <td>88.0</td>
      <td>143.0</td>
      <td>33.0</td>
      <td>132.0</td>
      <td>66.0</td>
      <td>22.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



Of course, there's also a `tail()` function to view the bottom 5 rows. You can pass the number of rows you want:


```python
large_df.tail(n=2)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>some_text</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>G</th>
      <th>H</th>
      <th>I</th>
      <th>...</th>
      <th>Q</th>
      <th>R</th>
      <th>S</th>
      <th>T</th>
      <th>U</th>
      <th>V</th>
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9998</th>
      <td>22.0</td>
      <td>33.0</td>
      <td>66.0</td>
      <td>Blabla</td>
      <td>121.0</td>
      <td>11.0</td>
      <td>110.0</td>
      <td>44.0</td>
      <td>NaN</td>
      <td>165.0</td>
      <td>...</td>
      <td>33.0</td>
      <td>22.0</td>
      <td>33.0</td>
      <td>66.0</td>
      <td>121.0</td>
      <td>11.0</td>
      <td>110.0</td>
      <td>44.0</td>
      <td>NaN</td>
      <td>165.0</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>33.0</td>
      <td>44.0</td>
      <td>77.0</td>
      <td>Blabla</td>
      <td>132.0</td>
      <td>22.0</td>
      <td>121.0</td>
      <td>55.0</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>44.0</td>
      <td>33.0</td>
      <td>44.0</td>
      <td>77.0</td>
      <td>132.0</td>
      <td>22.0</td>
      <td>121.0</td>
      <td>55.0</td>
      <td>11.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 27 columns</p>
</div>



The `info()` method prints out a summary of each column's contents:


```python
large_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 27 columns):
    A            8823 non-null float64
    B            8824 non-null float64
    C            8824 non-null float64
    some_text    10000 non-null object
    D            8824 non-null float64
    E            8822 non-null float64
    F            8824 non-null float64
    G            8824 non-null float64
    H            8822 non-null float64
    I            8823 non-null float64
    J            8823 non-null float64
    K            8822 non-null float64
    L            8824 non-null float64
    M            8824 non-null float64
    N            8822 non-null float64
    O            8824 non-null float64
    P            8824 non-null float64
    Q            8824 non-null float64
    R            8823 non-null float64
    S            8824 non-null float64
    T            8824 non-null float64
    U            8824 non-null float64
    V            8822 non-null float64
    W            8824 non-null float64
    X            8824 non-null float64
    Y            8822 non-null float64
    Z            8823 non-null float64
    dtypes: float64(26), object(1)
    memory usage: 2.1+ MB


Finally, the `describe()` method gives a nice overview of the main aggregated values over each column:
* `count`: number of non-null (not NaN) values
* `mean`: mean of non-null values
* `std`: [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation) of non-null values
* `min`: minimum of non-null values
* `25%`, `50%`, `75%`: 25th, 50th and 75th [percentile](https://en.wikipedia.org/wiki/Percentile) of non-null values
* `max`: maximum of non-null values


```python
large_df.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>G</th>
      <th>H</th>
      <th>I</th>
      <th>J</th>
      <th>...</th>
      <th>Q</th>
      <th>R</th>
      <th>S</th>
      <th>T</th>
      <th>U</th>
      <th>V</th>
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8823.000000</td>
      <td>8824.000000</td>
      <td>8824.000000</td>
      <td>8824.000000</td>
      <td>8822.000000</td>
      <td>8824.000000</td>
      <td>8824.000000</td>
      <td>8822.000000</td>
      <td>8823.000000</td>
      <td>8823.000000</td>
      <td>...</td>
      <td>8824.000000</td>
      <td>8823.000000</td>
      <td>8824.000000</td>
      <td>8824.000000</td>
      <td>8824.000000</td>
      <td>8822.000000</td>
      <td>8824.000000</td>
      <td>8824.000000</td>
      <td>8822.000000</td>
      <td>8823.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>87.977559</td>
      <td>87.972575</td>
      <td>87.987534</td>
      <td>88.012466</td>
      <td>87.983791</td>
      <td>88.007480</td>
      <td>87.977561</td>
      <td>88.000000</td>
      <td>88.022441</td>
      <td>88.022441</td>
      <td>...</td>
      <td>87.972575</td>
      <td>87.977559</td>
      <td>87.972575</td>
      <td>87.987534</td>
      <td>88.012466</td>
      <td>87.983791</td>
      <td>88.007480</td>
      <td>87.977561</td>
      <td>88.000000</td>
      <td>88.022441</td>
    </tr>
    <tr>
      <th>std</th>
      <td>47.535911</td>
      <td>47.535523</td>
      <td>47.521679</td>
      <td>47.521679</td>
      <td>47.535001</td>
      <td>47.519371</td>
      <td>47.529755</td>
      <td>47.536879</td>
      <td>47.535911</td>
      <td>47.535911</td>
      <td>...</td>
      <td>47.535523</td>
      <td>47.535911</td>
      <td>47.535523</td>
      <td>47.521679</td>
      <td>47.521679</td>
      <td>47.535001</td>
      <td>47.519371</td>
      <td>47.529755</td>
      <td>47.536879</td>
      <td>47.535911</td>
    </tr>
    <tr>
      <th>min</th>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>...</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>44.000000</td>
      <td>44.000000</td>
      <td>44.000000</td>
      <td>44.000000</td>
      <td>44.000000</td>
      <td>44.000000</td>
      <td>44.000000</td>
      <td>44.000000</td>
      <td>44.000000</td>
      <td>44.000000</td>
      <td>...</td>
      <td>44.000000</td>
      <td>44.000000</td>
      <td>44.000000</td>
      <td>44.000000</td>
      <td>44.000000</td>
      <td>44.000000</td>
      <td>44.000000</td>
      <td>44.000000</td>
      <td>44.000000</td>
      <td>44.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>88.000000</td>
      <td>88.000000</td>
      <td>88.000000</td>
      <td>88.000000</td>
      <td>88.000000</td>
      <td>88.000000</td>
      <td>88.000000</td>
      <td>88.000000</td>
      <td>88.000000</td>
      <td>88.000000</td>
      <td>...</td>
      <td>88.000000</td>
      <td>88.000000</td>
      <td>88.000000</td>
      <td>88.000000</td>
      <td>88.000000</td>
      <td>88.000000</td>
      <td>88.000000</td>
      <td>88.000000</td>
      <td>88.000000</td>
      <td>88.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>...</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>165.000000</td>
      <td>165.000000</td>
      <td>165.000000</td>
      <td>165.000000</td>
      <td>165.000000</td>
      <td>165.000000</td>
      <td>165.000000</td>
      <td>165.000000</td>
      <td>165.000000</td>
      <td>165.000000</td>
      <td>...</td>
      <td>165.000000</td>
      <td>165.000000</td>
      <td>165.000000</td>
      <td>165.000000</td>
      <td>165.000000</td>
      <td>165.000000</td>
      <td>165.000000</td>
      <td>165.000000</td>
      <td>165.000000</td>
      <td>165.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 26 columns</p>
</div>



# Saving & loading
Pandas can save `DataFrame`s to various backends, including file formats such as CSV, Excel, JSON, HTML and HDF5, or to a SQL database. Let's create a `DataFrame` to demonstrate this:


```python
my_df = pd.DataFrame(
    [["Biking", 68.5, 1985, np.nan], ["Dancing", 83.1, 1984, 3]],
    columns=["hobby", "weight", "birthyear", "children"],
    index=["alice", "bob"]
)
my_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hobby</th>
      <th>weight</th>
      <th>birthyear</th>
      <th>children</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>Biking</td>
      <td>68.5</td>
      <td>1985</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>Dancing</td>
      <td>83.1</td>
      <td>1984</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



## Saving
Let's save it to CSV, HTML and JSON:


```python
my_df.to_csv("my_df.csv")
my_df.to_html("my_df.html")
my_df.to_json("my_df.json")
```

Done! Let's take a peek at what was saved:


```python
for filename in ("my_df.csv", "my_df.html", "my_df.json"):
    print("#", filename)
    with open(filename, "rt") as f:
        print(f.read())
        print()

```

    # my_df.csv
    ,hobby,weight,birthyear,children
    alice,Biking,68.5,1985,
    bob,Dancing,83.1,1984,3.0
    
    
    # my_df.html
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>hobby</th>
          <th>weight</th>
          <th>birthyear</th>
          <th>children</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>alice</th>
          <td>Biking</td>
          <td>68.5</td>
          <td>1985</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>bob</th>
          <td>Dancing</td>
          <td>83.1</td>
          <td>1984</td>
          <td>3.0</td>
        </tr>
      </tbody>
    </table>
    
    # my_df.json
    {"hobby":{"alice":"Biking","bob":"Dancing"},"weight":{"alice":68.5,"bob":83.1},"birthyear":{"alice":1985,"bob":1984},"children":{"alice":null,"bob":3.0}}
    


Note that the index is saved as the first column (with no name) in a CSV file, as `<th>` tags in HTML and as keys in JSON.

Saving to other formats works very similarly, but some formats require extra libraries to be installed. For example, saving to Excel requires the openpyxl library:


```python
try:
    my_df.to_excel("my_df.xlsx", sheet_name='People')
except ImportError as e:
    print(e)
```

    No module named 'openpyxl'


## Loading
Now let's load our CSV file back into a `DataFrame`:


```python
my_df_loaded = pd.read_csv("my_df.csv", index_col=0)
my_df_loaded
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hobby</th>
      <th>weight</th>
      <th>birthyear</th>
      <th>children</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>Biking</td>
      <td>68.5</td>
      <td>1985</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>Dancing</td>
      <td>83.1</td>
      <td>1984</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



As you might guess, there are similar `read_json`, `read_html`, `read_excel` functions as well.  We can also read data straight from the Internet. For example, let's load the top 1,000 U.S. cities from GitHub:


```python
us_cities = None
try:
    csv_url = "https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv"
    us_cities = pd.read_csv(csv_url, index_col=0)
    us_cities = us_cities.head()
except IOError as e:
    print(e)
us_cities
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Population</th>
      <th>lat</th>
      <th>lon</th>
    </tr>
    <tr>
      <th>City</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Marysville</th>
      <td>Washington</td>
      <td>63269</td>
      <td>48.051764</td>
      <td>-122.177082</td>
    </tr>
    <tr>
      <th>Perris</th>
      <td>California</td>
      <td>72326</td>
      <td>33.782519</td>
      <td>-117.228648</td>
    </tr>
    <tr>
      <th>Cleveland</th>
      <td>Ohio</td>
      <td>390113</td>
      <td>41.499320</td>
      <td>-81.694361</td>
    </tr>
    <tr>
      <th>Worcester</th>
      <td>Massachusetts</td>
      <td>182544</td>
      <td>42.262593</td>
      <td>-71.802293</td>
    </tr>
    <tr>
      <th>Columbia</th>
      <td>South Carolina</td>
      <td>133358</td>
      <td>34.000710</td>
      <td>-81.034814</td>
    </tr>
  </tbody>
</table>
</div>



There are more options available, in particular regarding datetime format. Check out the [documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html) for more details.

# Combining `DataFrame`s

## SQL-like joins
One powerful feature of pandas is its ability to perform SQL-like joins on `DataFrame`s. Various types of joins are supported: inner joins, left/right outer joins and full joins. To illustrate this, let's start by creating a couple of simple `DataFrame`s:


```python
city_loc = pd.DataFrame(
    [
        ["CA", "San Francisco", 37.781334, -122.416728],
        ["NY", "New York", 40.705649, -74.008344],
        ["FL", "Miami", 25.791100, -80.320733],
        ["OH", "Cleveland", 41.473508, -81.739791],
        ["UT", "Salt Lake City", 40.755851, -111.896657]
    ], columns=["state", "city", "lat", "lng"])
city_loc
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>city</th>
      <th>lat</th>
      <th>lng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CA</td>
      <td>San Francisco</td>
      <td>37.781334</td>
      <td>-122.416728</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NY</td>
      <td>New York</td>
      <td>40.705649</td>
      <td>-74.008344</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FL</td>
      <td>Miami</td>
      <td>25.791100</td>
      <td>-80.320733</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OH</td>
      <td>Cleveland</td>
      <td>41.473508</td>
      <td>-81.739791</td>
    </tr>
    <tr>
      <th>4</th>
      <td>UT</td>
      <td>Salt Lake City</td>
      <td>40.755851</td>
      <td>-111.896657</td>
    </tr>
  </tbody>
</table>
</div>




```python
city_pop = pd.DataFrame(
    [
        [808976, "San Francisco", "California"],
        [8363710, "New York", "New-York"],
        [413201, "Miami", "Florida"],
        [2242193, "Houston", "Texas"]
    ], index=[3,4,5,6], columns=["population", "city", "state"])
city_pop
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>population</th>
      <th>city</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>808976</td>
      <td>San Francisco</td>
      <td>California</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8363710</td>
      <td>New York</td>
      <td>New-York</td>
    </tr>
    <tr>
      <th>5</th>
      <td>413201</td>
      <td>Miami</td>
      <td>Florida</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2242193</td>
      <td>Houston</td>
      <td>Texas</td>
    </tr>
  </tbody>
</table>
</div>



Now let's join these `DataFrame`s using the `merge()` function:


```python
pd.merge(left=city_loc, right=city_pop, on="city")
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state_x</th>
      <th>city</th>
      <th>lat</th>
      <th>lng</th>
      <th>population</th>
      <th>state_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CA</td>
      <td>San Francisco</td>
      <td>37.781334</td>
      <td>-122.416728</td>
      <td>808976</td>
      <td>California</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NY</td>
      <td>New York</td>
      <td>40.705649</td>
      <td>-74.008344</td>
      <td>8363710</td>
      <td>New-York</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FL</td>
      <td>Miami</td>
      <td>25.791100</td>
      <td>-80.320733</td>
      <td>413201</td>
      <td>Florida</td>
    </tr>
  </tbody>
</table>
</div>



Note that both `DataFrame`s have a column named `state`, so in the result they got renamed to `state_x` and `state_y`.

Also, note that Cleveland, Salt Lake City and Houston were dropped because they don't exist in *both* `DataFrame`s. This is the equivalent of a SQL `INNER JOIN`. If you want a `FULL OUTER JOIN`, where no city gets dropped and `NaN` values are added, you must specify `how="outer"`:


```python
all_cities = pd.merge(left=city_loc, right=city_pop, on="city", how="outer")
all_cities
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state_x</th>
      <th>city</th>
      <th>lat</th>
      <th>lng</th>
      <th>population</th>
      <th>state_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CA</td>
      <td>San Francisco</td>
      <td>37.781334</td>
      <td>-122.416728</td>
      <td>808976.0</td>
      <td>California</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NY</td>
      <td>New York</td>
      <td>40.705649</td>
      <td>-74.008344</td>
      <td>8363710.0</td>
      <td>New-York</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FL</td>
      <td>Miami</td>
      <td>25.791100</td>
      <td>-80.320733</td>
      <td>413201.0</td>
      <td>Florida</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OH</td>
      <td>Cleveland</td>
      <td>41.473508</td>
      <td>-81.739791</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>UT</td>
      <td>Salt Lake City</td>
      <td>40.755851</td>
      <td>-111.896657</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>Houston</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2242193.0</td>
      <td>Texas</td>
    </tr>
  </tbody>
</table>
</div>



Of course, `LEFT OUTER JOIN` is also available by setting `how="left"`: only the cities present in the left `DataFrame` end up in the result. Similarly, with `how="right"` only cities in the right `DataFrame` appear in the result. For example:


```python
pd.merge(left=city_loc, right=city_pop, on="city", how="right")
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state_x</th>
      <th>city</th>
      <th>lat</th>
      <th>lng</th>
      <th>population</th>
      <th>state_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CA</td>
      <td>San Francisco</td>
      <td>37.781334</td>
      <td>-122.416728</td>
      <td>808976</td>
      <td>California</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NY</td>
      <td>New York</td>
      <td>40.705649</td>
      <td>-74.008344</td>
      <td>8363710</td>
      <td>New-York</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FL</td>
      <td>Miami</td>
      <td>25.791100</td>
      <td>-80.320733</td>
      <td>413201</td>
      <td>Florida</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>Houston</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2242193</td>
      <td>Texas</td>
    </tr>
  </tbody>
</table>
</div>



If the key to join on is actually in one (or both) `DataFrame`'s index, you must use `left_index=True` and/or `right_index=True`. If the key column names differ, you must use `left_on` and `right_on`. For example:


```python
city_pop2 = city_pop.copy()
city_pop2.columns = ["population", "name", "state"]
pd.merge(left=city_loc, right=city_pop2, left_on="city", right_on="name")
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state_x</th>
      <th>city</th>
      <th>lat</th>
      <th>lng</th>
      <th>population</th>
      <th>name</th>
      <th>state_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CA</td>
      <td>San Francisco</td>
      <td>37.781334</td>
      <td>-122.416728</td>
      <td>808976</td>
      <td>San Francisco</td>
      <td>California</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NY</td>
      <td>New York</td>
      <td>40.705649</td>
      <td>-74.008344</td>
      <td>8363710</td>
      <td>New York</td>
      <td>New-York</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FL</td>
      <td>Miami</td>
      <td>25.791100</td>
      <td>-80.320733</td>
      <td>413201</td>
      <td>Miami</td>
      <td>Florida</td>
    </tr>
  </tbody>
</table>
</div>



## Concatenation
Rather than joining `DataFrame`s, we may just want to concatenate them. That's what `concat()` is for:


```python
result_concat = pd.concat([city_loc, city_pop])
result_concat
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>lat</th>
      <th>lng</th>
      <th>population</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>San Francisco</td>
      <td>37.781334</td>
      <td>-122.416728</td>
      <td>NaN</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>New York</td>
      <td>40.705649</td>
      <td>-74.008344</td>
      <td>NaN</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Miami</td>
      <td>25.791100</td>
      <td>-80.320733</td>
      <td>NaN</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cleveland</td>
      <td>41.473508</td>
      <td>-81.739791</td>
      <td>NaN</td>
      <td>OH</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Salt Lake City</td>
      <td>40.755851</td>
      <td>-111.896657</td>
      <td>NaN</td>
      <td>UT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>San Francisco</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>808976.0</td>
      <td>California</td>
    </tr>
    <tr>
      <th>4</th>
      <td>New York</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8363710.0</td>
      <td>New-York</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Miami</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>413201.0</td>
      <td>Florida</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Houston</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2242193.0</td>
      <td>Texas</td>
    </tr>
  </tbody>
</table>
</div>



Note that this operation aligned the data horizontally (by columns) but not vertically (by rows). In this example, we end up with multiple rows having the same index (e.g. 3). Pandas handles this rather gracefully:


```python
result_concat.loc[3]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>lat</th>
      <th>lng</th>
      <th>population</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Cleveland</td>
      <td>41.473508</td>
      <td>-81.739791</td>
      <td>NaN</td>
      <td>OH</td>
    </tr>
    <tr>
      <th>3</th>
      <td>San Francisco</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>808976.0</td>
      <td>California</td>
    </tr>
  </tbody>
</table>
</div>



Or you can tell pandas to just ignore the index:


```python
pd.concat([city_loc, city_pop], ignore_index=True)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>lat</th>
      <th>lng</th>
      <th>population</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>San Francisco</td>
      <td>37.781334</td>
      <td>-122.416728</td>
      <td>NaN</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>New York</td>
      <td>40.705649</td>
      <td>-74.008344</td>
      <td>NaN</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Miami</td>
      <td>25.791100</td>
      <td>-80.320733</td>
      <td>NaN</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cleveland</td>
      <td>41.473508</td>
      <td>-81.739791</td>
      <td>NaN</td>
      <td>OH</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Salt Lake City</td>
      <td>40.755851</td>
      <td>-111.896657</td>
      <td>NaN</td>
      <td>UT</td>
    </tr>
    <tr>
      <th>5</th>
      <td>San Francisco</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>808976.0</td>
      <td>California</td>
    </tr>
    <tr>
      <th>6</th>
      <td>New York</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8363710.0</td>
      <td>New-York</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Miami</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>413201.0</td>
      <td>Florida</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Houston</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2242193.0</td>
      <td>Texas</td>
    </tr>
  </tbody>
</table>
</div>



Notice that when a column does not exist in a `DataFrame`, it acts as if it was filled with `NaN` values. If we set `join="inner"`, then only columns that exist in *both* `DataFrame`s are returned:


```python
pd.concat([city_loc, city_pop], join="inner")
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CA</td>
      <td>San Francisco</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NY</td>
      <td>New York</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FL</td>
      <td>Miami</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OH</td>
      <td>Cleveland</td>
    </tr>
    <tr>
      <th>4</th>
      <td>UT</td>
      <td>Salt Lake City</td>
    </tr>
    <tr>
      <th>3</th>
      <td>California</td>
      <td>San Francisco</td>
    </tr>
    <tr>
      <th>4</th>
      <td>New-York</td>
      <td>New York</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Florida</td>
      <td>Miami</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Texas</td>
      <td>Houston</td>
    </tr>
  </tbody>
</table>
</div>



You can concatenate `DataFrame`s horizontally instead of vertically by setting `axis=1`:


```python
pd.concat([city_loc, city_pop], axis=1)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>city</th>
      <th>lat</th>
      <th>lng</th>
      <th>population</th>
      <th>city</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CA</td>
      <td>San Francisco</td>
      <td>37.781334</td>
      <td>-122.416728</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NY</td>
      <td>New York</td>
      <td>40.705649</td>
      <td>-74.008344</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FL</td>
      <td>Miami</td>
      <td>25.791100</td>
      <td>-80.320733</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OH</td>
      <td>Cleveland</td>
      <td>41.473508</td>
      <td>-81.739791</td>
      <td>808976.0</td>
      <td>San Francisco</td>
      <td>California</td>
    </tr>
    <tr>
      <th>4</th>
      <td>UT</td>
      <td>Salt Lake City</td>
      <td>40.755851</td>
      <td>-111.896657</td>
      <td>8363710.0</td>
      <td>New York</td>
      <td>New-York</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>413201.0</td>
      <td>Miami</td>
      <td>Florida</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2242193.0</td>
      <td>Houston</td>
      <td>Texas</td>
    </tr>
  </tbody>
</table>
</div>



In this case it really does not make much sense because the indices do not align well (e.g. Cleveland and San Francisco end up on the same row, because they shared the index label `3`). So let's reindex the `DataFrame`s by city name before concatenating:


```python
pd.concat([city_loc.set_index("city"), city_pop.set_index("city")], axis=1)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>lat</th>
      <th>lng</th>
      <th>population</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cleveland</th>
      <td>OH</td>
      <td>41.473508</td>
      <td>-81.739791</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Houston</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2242193.0</td>
      <td>Texas</td>
    </tr>
    <tr>
      <th>Miami</th>
      <td>FL</td>
      <td>25.791100</td>
      <td>-80.320733</td>
      <td>413201.0</td>
      <td>Florida</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>NY</td>
      <td>40.705649</td>
      <td>-74.008344</td>
      <td>8363710.0</td>
      <td>New-York</td>
    </tr>
    <tr>
      <th>Salt Lake City</th>
      <td>UT</td>
      <td>40.755851</td>
      <td>-111.896657</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>San Francisco</th>
      <td>CA</td>
      <td>37.781334</td>
      <td>-122.416728</td>
      <td>808976.0</td>
      <td>California</td>
    </tr>
  </tbody>
</table>
</div>



This looks a lot like a `FULL OUTER JOIN`, except that the `state` columns were not renamed to `state_x` and `state_y`, and the `city` column is now the index.

# Categories
It is quite frequent to have values that represent categories, for example `1` for female and `2` for male, or `"A"` for Good, `"B"` for Average, `"C"` for Bad. These categorical values can be hard to read and cumbersome to handle, but fortunately pandas makes it easy. To illustrate this, let's take the `city_pop` `DataFrame` we created earlier, and add a column that represents a category:


```python
city_eco = city_pop.copy()
city_eco["eco_code"] = [17, 17, 34, 20]
city_eco
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>population</th>
      <th>city</th>
      <th>state</th>
      <th>eco_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>808976</td>
      <td>San Francisco</td>
      <td>California</td>
      <td>17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8363710</td>
      <td>New York</td>
      <td>New-York</td>
      <td>17</td>
    </tr>
    <tr>
      <th>5</th>
      <td>413201</td>
      <td>Miami</td>
      <td>Florida</td>
      <td>34</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2242193</td>
      <td>Houston</td>
      <td>Texas</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>



Right now the `eco_code` column is full of apparently meaningless codes. Let's fix that. First, we will create a new categorical column based on the `eco_code`s:


```python
city_eco["economy"] = city_eco["eco_code"].astype('category')
city_eco["economy"].cat.categories
```




    Int64Index([17, 20, 34], dtype='int64')



Now we can give each category a meaningful name:


```python
city_eco["economy"].cat.categories = ["Finance", "Energy", "Tourism"]
city_eco
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>population</th>
      <th>city</th>
      <th>state</th>
      <th>eco_code</th>
      <th>economy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>808976</td>
      <td>San Francisco</td>
      <td>California</td>
      <td>17</td>
      <td>Finance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8363710</td>
      <td>New York</td>
      <td>New-York</td>
      <td>17</td>
      <td>Finance</td>
    </tr>
    <tr>
      <th>5</th>
      <td>413201</td>
      <td>Miami</td>
      <td>Florida</td>
      <td>34</td>
      <td>Tourism</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2242193</td>
      <td>Houston</td>
      <td>Texas</td>
      <td>20</td>
      <td>Energy</td>
    </tr>
  </tbody>
</table>
</div>



Note that categorical values are sorted according to their categorical order, *not* their alphabetical order:


```python
city_eco.sort_values(by="economy", ascending=False)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>population</th>
      <th>city</th>
      <th>state</th>
      <th>eco_code</th>
      <th>economy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>413201</td>
      <td>Miami</td>
      <td>Florida</td>
      <td>34</td>
      <td>Tourism</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2242193</td>
      <td>Houston</td>
      <td>Texas</td>
      <td>20</td>
      <td>Energy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8363710</td>
      <td>New York</td>
      <td>New-York</td>
      <td>17</td>
      <td>Finance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>808976</td>
      <td>San Francisco</td>
      <td>California</td>
      <td>17</td>
      <td>Finance</td>
    </tr>
  </tbody>
</table>
</div>



# What's next?
As you probably noticed by now, pandas is quite a large library with *many* features. Although we went through the most important features, there is still a lot to discover. Probably the best way to learn more is to get your hands dirty with some real-life data. It is also a good idea to go through pandas' excellent [documentation](https://pandas.pydata.org/pandas-docs/stable/index.html), in particular the [Cookbook](https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html).


```python

```
