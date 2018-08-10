#include <chrono>
#include <string>
#include <iostream>

namespace Util
{
    //http://stackoverflow.com/questions/16518705/problems-with-stdchrono/16519154#16519154
    //steady_clock is only available in gcc > 4.7, so use monotonic_clock instead.
#if __cplusplus < 201103L
    typedef std::chrono::monotonic_clock scm_steady_clock;
#else
    typedef std::chrono::steady_clock scm_steady_clock;
#endif

    class ScopedTimer
    {
    public:
        ScopedTimer();
        explicit ScopedTimer( const std::string& name_of_function_being_timed, std::ostream& output_stream = std::cout );
        explicit ScopedTimer( std::ostream& output_stream, const std::string& name_of_function_being_timed = "" );
        ~ScopedTimer();
    private:
        std::string name_of_function_being_timed_;
        std::ostream& output_stream_;
        
        scm_steady_clock::time_point start_time_;
        scm_steady_clock::time_point end_time_;
    };

    //No delegating constructors in GCC 4.6.3
    inline ScopedTimer::ScopedTimer()
        : name_of_function_being_timed_{}
        , output_stream_( std::cout )
        , start_time_{ scm_steady_clock::now() }
        , end_time_{}
    {
    }
    
    //No delegating constructors in GCC 4.6.3
    inline ScopedTimer::ScopedTimer( const std::string& name_of_function_being_timed, std::ostream& output_stream )
        : name_of_function_being_timed_{ name_of_function_being_timed }
        , output_stream_( output_stream )
        , start_time_{ scm_steady_clock::now() }
        , end_time_{}
    {
    }
    
    //No delegating constructors in GCC 4.6.3
    inline ScopedTimer::ScopedTimer( std::ostream& output_stream, const std::string& name_of_function_being_timed )
        : name_of_function_being_timed_{ name_of_function_being_timed }
        , output_stream_( output_stream )
        , start_time_{ scm_steady_clock::now() }
        , end_time_{}
    {
    }

    inline ScopedTimer::~ScopedTimer()
    {
        end_time_ = scm_steady_clock::now();
        const std::chrono::duration<double> elapsed_time =  end_time_ - start_time_;
        output_stream_ << ( name_of_function_being_timed_.empty() ? "" : name_of_function_being_timed_ + ": " ) <<
            "elapsed time: " << elapsed_time.count() << " seconds." << '\n';
    }
} //namespace Util
