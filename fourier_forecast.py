import pandas
import numpy
from sklearn.linear_model import LassoCV, LinearRegression


class FourierForecast:
    """
    is a class that can be used to fit seasonal time series data by decomposing the signal and removing the noise
    this can be used to predict values into the future
    additional features such as public holidays, sale days can also be added and fitted as well
    """
    coefs = None
    poly_coefs = None
    signals = None
    datetime_start = None
    datetime_freq = None
    model = None
    check_columns = None
    seasonality = None
    trend = None
    noise = None

    def __init__(self, train_size: int = None, fourier_order: int = 5):
        """
        :param train_size: max number of rows used to fit data
        :param fourier_order: represents granularity of seasonality to fit
        higher values will capture more levels of seasonality but risks over-fitting
        """

        if (isinstance(train_size, int) & (train_size != 0)) | isinstance(train_size, type(None)):
            self.train_size = train_size
        else:
            raise TypeError('train_size must be None or a non-zero integer')

        self.fourier_order = int(fourier_order)

    @staticmethod
    def create_signals(n: int, t: numpy.array, y: numpy.array) -> numpy.array:
        """
        create and filter signal based on random n
        :param n: number of periods to fit out of the total number of periods
        :param t: array of datetime like values
        :param y: values to be fitted
        :return: the cleaned signals that best fit y
        """

        start = 0
        ynew = y[start: n]
        tnew = t[start: n]

        # find and remove trend
        poly_coefs = numpy.polyfit(tnew, ynew, deg=1)
        trend = numpy.poly1d(poly_coefs)(tnew)
        ynew = ynew - trend

        # fft
        yhat = numpy.fft.rfft(ynew)
        # power spectral density
        psd = numpy.abs(yhat) / n
        # frequencies, dt is just 1 unit
        xf = numpy.fft.rfftfreq(n)

        # create filter based on psd
        filter_value = numpy.mean(psd) + 2 * numpy.std(psd, ddof=0)
        flter = psd >= filter_value

        # filter freqs, amps and phases
        frequencies = numpy.abs(xf[flter])
        amplitudes = numpy.abs(yhat) * 2 / n
        amplitudes = amplitudes[flter]
        phases = numpy.arctan2(yhat[flter].imag, yhat[flter].real) + numpy.pi / 2
        # adjust phases to t=0
        phases = phases - 2 * numpy.pi * frequencies * start

        return numpy.array([frequencies, amplitudes, phases]).T

    @staticmethod
    def build_signal(signals: numpy.array, t: numpy.array) -> numpy.array:
        """
        create one composite signal from individual signals
        :param signals: individual signals that will be summed
        :param t: array of datetime like values
        :return: a composite signal
        """

        new_signal = numpy.zeros(t.size, dtype=numpy.float64)
        for s in signals:
            # 0 = freq, 1 = amplitude, 2 = phase
            new_signal += s[1] * numpy.sin(s[2] + 2 * numpy.pi * s[0] * t)

        return new_signal

    def find_best_signal(self, ns: numpy.array, y_adjust: numpy.array, t: numpy.array) -> (numpy.array, numpy.float64):
        """
        find the signal (out of the cleaned signals created) that best fits y_adjust by minimising the mse
        for each number of periods (for n in ns),
        cleaned signals are created, and for each signal of all ns the one that minimises mse the best is the one
        that is returned
        :param ns: each n in ns represents the number of periods to fit out of the total length of y
        :param y_adjust: y after the previous fourier level signal has been fitted then removed
        :param t: array of datetime like values
        :return: the single best signal that fits y adjust and mse_inner - the error term
        """

        mse_inner = numpy.inf
        for n in ns:
            # create and filter signal
            signals = self.create_signals(n, t, y_adjust)

            for sn in range(len(signals)):
                # change amp to 0
                signal_to_build = signals[sn]
                signal_to_build[1] = numpy.float(1)
                new_signal = self.build_signal([signal_to_build], t)

                x = numpy.array([new_signal], dtype=numpy.float64).T
                model = LinearRegression(fit_intercept=True, positive=True)
                model.fit(x, y_adjust)
                amp = model.coef_[0]
                new_signal = new_signal * amp

                # create and add new trend
                y_de_seasoned = y_adjust - new_signal
                poly_coefs = numpy.polyfit(t, y_de_seasoned, deg=1)
                trend = numpy.poly1d(poly_coefs)(t)
                new_signal += trend

                # evaluate fit
                mse_new = numpy.mean((new_signal - y_adjust) ** 2)
                if mse_new < mse_inner:
                    mse_inner = mse_new
                    best_signal = numpy.array([signal_to_build])
                    # change amp to coef
                    best_signal[0][1] = amp

        return best_signal, mse_inner

    def fit_seasonality(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """
        calculate seasonality & trends with fast fourier transformation
        it will fit the best signal then remove it to fit the residual signal
        and so on for the number of fourier orders given
        if the next fourier order fit does not improve the overall fit then it will stop fitting further levels
        :param df: dataframe with ds and y columns
        :return: df with column for each fourier order signal
        """

        max_n = len(df)
        min_n = (max_n + 2) // 2
        ns = numpy.arange(min_n, max_n + 1)
        y_adjust = numpy.array(df['y'])
        t = numpy.array(df['ds'])
        all_signals = []
        frequencies = []
        mse = numpy.inf

        # find signal that best fits y then subtract and repeat to find signal that fits the remaining y
        for i in range(self.fourier_order):

            best_signal, mse_inner = self.find_best_signal(ns, y_adjust, t)

            # if frequency has already been used then end
            if best_signal[0][0] in frequencies:
                break
            else:
                frequencies.append(best_signal[0][0])

            # if mse has improved 10% then continue else end to avoid over fitting
            if mse_inner < mse / 1.1:
                mse = mse_inner
                all_signals.extend(best_signal)
                # remove best signal from y, so we can fit a new signal to the remainder
                y_adjust -= self.build_signal(best_signal, t)
            else:
                break

        self.signals = numpy.array(all_signals)

        # find global trend
        poly_coefs = numpy.polyfit(t, y_adjust, deg=1)
        self.poly_coefs = poly_coefs
        global_trend = numpy.poly1d(poly_coefs)(t)
        df['trend'] = global_trend
        self.trend = global_trend

        # build overall seasonality
        self.seasonality = self.build_signal(self.signals, t)

        # add individual signals to dataframe
        for s in self.signals:
            col = str(round(1 / s[0], 6))
            # 0 = freq, 1 = amplitude, 2 = phase
            sig = s[1] * numpy.sin(s[2] + 2 * numpy.pi * s[0] * t)
            df[col] = sig

        self.noise = numpy.array(df['y'] - self.trend - self.seasonality)

        return df

    def fit(self, df: pandas.DataFrame):
        """
        fits model
        :param df: pandas.DataFrame with a datetime column labelled 'ds' and values column labelled 'y'
        can optionally contain additional columns such as holidays to improve fit
        """

        # validate input is dataframe
        if not isinstance(df, pandas.DataFrame):
            raise TypeError('df must be a pandas dataframe')

        # validate columns of dateframe contain ds & y
        if not (('ds' in df.columns) & ('y' in df.columns)):
            raise ValueError('df must contain columns labelled ds and y')

        # save columns to validate for prediction df
        self.check_columns = [c for c in df.columns if c != 'y']

        # assign datetime type and sort dataframe
        df = (df
              .assign(ds=lambda x: pandas.to_datetime(x['ds']))
              .sort_values(by='ds')
              )

        # if train size is defined take tail
        if self.train_size is not None:
            df = df.tail(self.train_size)

        # take frequency to validate prediction df and check for missing datetimes
        freq = pandas.infer_freq(df['ds'])
        self.datetime_freq = freq
        dr = pandas.date_range(df['ds'].min(), df['ds'].max(), freq=freq)
        if len(df) < len(dr):
            raise ValueError('dataframe is missing datetime entries')
        if len(df) > len(dr):
            raise ValueError('dataframe has too many datetime entries')

        # convert ds to numeric array
        self.datetime_start = df['ds'].min()
        df['ds'] = numpy.arange(len(df))

        # fit seasonality and trend
        df_tofit = self.fit_seasonality(df)

        # fit model with Lasso to help avoid overfitting signal and fit additional features if provided
        x = df_tofit.drop(columns=['ds', 'y'])
        model = LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        fit_intercept=True, positive=False,
                        max_iter=1000, cv=5,
                        )
        model.fit(x, df_tofit['y'])
        self.model = model

        # make coefficients
        coefs = dict(zip(x.columns, model.coef_))
        coefs = {c: coefs[c] for c in coefs if coefs[c] != float(0)}
        coefs['intercept'] = model.intercept_
        self.coefs = coefs

    def predict(self, df: pandas.DataFrame) -> numpy.array:
        """
        creates predictions based on given datetimes
        :param df: pandas.DataFrame with a datetime column labelled 'ds' of same freq as dataframe that was fitted
        should not contain 'y' column
        :return: predicted values
        """

        # check input in dataframe
        if not isinstance(df, pandas.DataFrame):
            raise TypeError('df must be a pandas dataframe')

        if 'ds' not in df.columns:
            raise ValueError('df must contain columns labelled ds and y')

        # checking extra columns have not been added that were not used for fitting
        extra_cols = [c for c in df.columns if c not in self.check_columns]
        if len(extra_cols):
            raise ValueError(f'dataframe contains extra columns {extra_cols}')

        # assign datetime type and sort dataframe
        df = (df
              .assign(ds=lambda u: pandas.to_datetime(u['ds']))
              .sort_values(by='ds')
              )

        # check freq matches fitted datetimes
        if pandas.infer_freq(df['ds']) != self.datetime_freq:
            raise ValueError('new datetime array does not match freq of fitted array')

        # create new dataframe for making predictions
        new_df = pandas.DataFrame()
        start = numpy.min([self.datetime_start, df['ds'].min()])
        new_df['ds'] = pandas.date_range(start, df['ds'].max(), freq=self.datetime_freq)

        # remember pred start for returning final preds
        index_pred_start = new_df['ds'][new_df['ds'] == df['ds'].min()].index[0]

        # find fit start to align signals
        index_fit_start = new_df['ds'][new_df['ds'] == self.datetime_start].index[0]
        new_df['ds'] = new_df.index - index_fit_start

        # trend
        new_df['trend'] = numpy.poly1d(self.poly_coefs)(new_df['ds'])

        # build signals
        for s in self.signals:
            col = str(round(1 / s[0], 6))
            # 0 = freq, 1 = amplitude, 2 = phase
            new_df[col] = s[1] * numpy.sin(s[2] + 2 * numpy.pi * s[0] * new_df['ds'])

        # filter after pred start
        x = (new_df[new_df['ds'] >= index_pred_start]
             .copy(deep=True)
             .drop(columns=['ds'])
             )

        return self.model.predict(x)
