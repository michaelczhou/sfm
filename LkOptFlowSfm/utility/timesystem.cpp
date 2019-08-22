#include "timesystem.h"
//#include <glog/logging.h>
// timesystem
///////////////////////////////////////////////////////////////////////////////
// 局部函数声明
///////////////////////////////////////////////////////////////////////////////

/*变量定义*/
const static double gpst0[] = {1980, 1, 6, 0, 0, 0}; /* gps time reference */
static double leaps[100 + 1][7] = {                  /* leap seconds (y,m,d,h,m,s,utc-gpst) */
                                   {2017, 1, 1, 0, 0, 0, -18},
                                   {2015, 7, 1, 0, 0, 0, -17},
                                   {2012, 7, 1, 0, 0, 0, -16},
                                   {2009, 1, 1, 0, 0, 0, -15},
                                   {2006, 1, 1, 0, 0, 0, -14},
                                   {1999, 1, 1, 0, 0, 0, -13},
                                   {1997, 7, 1, 0, 0, 0, -12},
                                   {1996, 1, 1, 0, 0, 0, -11},
                                   {1994, 7, 1, 0, 0, 0, -10},
                                   {1993, 7, 1, 0, 0, 0, -9},
                                   {1992, 7, 1, 0, 0, 0, -8},
                                   {1991, 1, 1, 0, 0, 0, -7},
                                   {1990, 1, 1, 0, 0, 0, -6},
                                   {1988, 1, 1, 0, 0, 0, -5},
                                   {1985, 7, 1, 0, 0, 0, -4},
                                   {1983, 7, 1, 0, 0, 0, -3},
                                   {1982, 7, 1, 0, 0, 0, -2},
                                   {1981, 7, 1, 0, 0, 0, -1},
                                   {0}};

/* add time --------------------------------------------------------------------
* add time to gtime_t struct
* args   : gtime_t t        I   gtime_t struct
*          double sec       I   time to add (s)
* return : gtime_t struct (t+sec)
*-----------------------------------------------------------------------------*/
gtime_t timeadd(gtime_t t, double sec)
{
    double tt;

    t.sec += sec;
    tt = floor(t.sec);
    t.time += (int)tt;
    t.sec -= tt;
    return t;
}
/* time difference -------------------------------------------------------------
* difference between gtime_t structs
* args   : gtime_t t1,t2    I   gtime_t structs
* return : time difference (t1-t2) (s)
*-----------------------------------------------------------------------------*/
double timediff(gtime_t t1, gtime_t t2)
{
    return difftime(t1.time, t2.time) + t1.sec - t2.sec;
}
/* convert calendar day/time to time -------------------------------------------
* convert calendar day/time to gtime_t struct
* args   : double *ep       I   day/time {year,month,day,hour,min,sec}
* return : gtime_t struct
* notes  : proper in 1970-2037 or 1970-2099 (64bit time_t)
*-----------------------------------------------------------------------------*/
gtime_t epoch2time(const double *ep)
{
    const int doy[] = {1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335};
    gtime_t time = {0};
    int days, sec, year = (int)ep[0], mon = (int)ep[1], day = (int)ep[2];

    if (year < 1970 || 2099 < year || mon < 1 || 12 < mon)
        return time;

    /* leap year if year%4==0 in 1901-2099 */
    days = (year - 1970) * 365 + (year - 1969) / 4 + doy[mon - 1] + day - 2 + (year % 4 == 0 && mon >= 3 ? 1 : 0);
    sec = (int)floor(ep[5]);
    time.time = (time_t)days * 86400 + (int)ep[3] * 3600 + (int)ep[4] * 60 + sec;
    time.sec = ep[5] - sec;
    return time;
}
/* utc to gpstime --------------------------------------------------------------
* convert utc to gpstime considering leap seconds
* args   : gtime_t t        I   time expressed in utc
* return : time expressed in gpstime
* notes  : ignore slight time offset under 100 ns
*-----------------------------------------------------------------------------*/
gtime_t utc2gpst(gtime_t t)
{
    int i;

    for (i = 0; leaps[i][0] > 0; i++)
    {
        if (timediff(t, epoch2time(leaps[i])) >= 0.0)
            return timeadd(t, -leaps[i][6]);
    }
    return t;
}
/* time to gps time ------------------------------------------------------------
* convert gtime_t struct to week and tow in gps time
* args   : gtime_t t        I   gtime_t struct
*          int    *week     IO  week number in gps time (NULL: no output)
* return : time of week in gps time (s)
*-----------------------------------------------------------------------------*/
double time2gpst(gtime_t t, int *week)
{
    gtime_t t0 = epoch2time(gpst0);
    time_t sec = t.time - t0.time;
    int w = (int)(sec / (86400 * 7));

    if (week)
        *week = w;
    return (double)(sec - w * 86400 * 7) + t.sec;
}

void time2WS(uint64_t t, uint16_t &week, uint32_t &ms)
{
    // 不满一秒部分
    uint32_t t0 = t % 1000;

    // 整秒部分
    uint64_t t1 = t - t0;

    // 整周部分
    week = static_cast<uint16_t>(t1 / (604800 * 1000));

    // 毫秒部分
    ms = static_cast<uint32_t>((t - week * 604800 * 1000));
}

gtime_t gpst2time(int week, double sec)
{
    gtime_t t = epoch2time(gpst0);

    if (sec < -1E9 || 1E9 < sec)
        sec = 0.0;
    t.time += 86400 * 7 * week + (int)sec;
    t.sec = sec - (int)sec;
    return t;
}

gtime_t gpst2utc(gtime_t t)
{
    gtime_t tu;
    int i;

    for (i = 0; i < (int)sizeof(leaps) / (int)sizeof(*leaps); i++)
    {
        tu = timeadd(t, leaps[i][6]);
        if (timediff(tu, epoch2time(leaps[i])) >= 0.0)
            return tu;
    }
    return t;
}

/*系统时转换为GPS周和周内秒----------------------------------------------------
* 输入/输出   : uint64_t  time      I   系统时间,单位为us
*               uint16_t  week     IO    GPS周
*               uint32_t  tow_ms   IO    GPS周内秒
-----------------------------------------------------------------------------*/
void weeksconvert(uint64_t time, uint16_t &week, uint32_t &tow_ms)
{
    int week_temp = 0;
    double secofweek = 0.0;
    gtime_t temp_time;
    temp_time.time = time / 1000000;
    temp_time.sec = ((double)time - temp_time.time * 1000000) / 1000000.0;
    secofweek = time2gpst(utc2gpst(temp_time), &week_temp); //(单位 s)
    week = (uint16_t)week_temp;
    tow_ms = (uint32_t)round(secofweek * 1000); // 此处需要四舍五入
}

double Unix2GPST(uint64_t time, uint16_t *week, bool b10ms)
{
    using namespace std;
    uint16_t week0 = 0;
    uint32_t sec0 = 0;
    weeksconvert(time, week0, sec0);
    if (week != NULL)
        *week = week0;
    double sec = 0;
    // 是否四舍五入到10ms
    if (b10ms)
        sec = round(sec0 / 10.0) * 0.01;
    else
        sec = sec0 / 1000.0;
    return sec;
}

long long GPSWS2Unix(const uint16_t week, const double sec)
{
    return gpst2utc(gpst2time(week, sec)).ToUnix_us();
}

long long SecOfDay2Unix(const long long rt, const double secofday)
{
    uint16_t week = 0;
    uint32_t sec = 0;
    time2WS(rt, week, sec);
    int day = sec / (86400 * 1000);
    long long timestamp_us = week * (86400LL * 1000 * 7) * 1000 + day * (86400LL * 1000) * 1000 + HHHMMSS2Sec(secofday) * 1000 * 1000;
    return timestamp_us;
}

double HHHMMSS2Sec(const double t)
{
    int hms = int(t);
    double ms = t - hms;
    int h = hms / 10000;
    int min = (hms - h * 10000) / 100;
    int sec = (hms - h * 10000 - min * 100);
    double t1 = h * 3600 + min * 60 + sec + ms;

    return t1;
}